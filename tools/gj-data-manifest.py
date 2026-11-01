#!/usr/bin/env python3
"""Build a source-grouped GJ 4K manifest, without training or rewriting datasets.

Standard library only. Run from the project root:
    python3 tools/gj-data-manifest.py
The output directory must not already contain manifest.json or REPORT.md.
Import build_manifest()/parse_chart()/beat_at_ms() for downstream data builders.
"""
from __future__ import annotations

import argparse
import ast
import bisect
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import unicodedata
import zipfile

STYLES = ("tech", "jack", "speed", "stamina")
SPLITS = ("train", "val", "test")
UNKNOWN = {"", "unknown", "none", "na", "untitled"}
GENERIC_AUDIO = UNKNOWN | {"audio", "song", "music", "track", "0", "1"}

AUDIT_FINDINGS = [
    {"id": "window_random_split", "evidence": "tools/train-diffusion.py:302-309; tools/train-diffusion32c.py:340-354",
     "finding": "窗口 randperm 后切 train/val，没有 chart/song 分组；同曲、重叠窗口可被分到两侧。没有旧窗口来源，不能证明某次历史切分的实际同曲交集为零。"},
    {"id": "timing_section_and_tempo", "evidence": "tools/diff-data-true.py:19-30,69-90",
     "finding": "timing() 扫全文件而非 TimingPoints 节，只取首个正 beatLength；HitObjects 可被误读为 timing。变速检测读最后字段 effects 而非第7字段 uninherited，未支持分段变速；量化忽略红线 offset，span 又按首尾差计算，可能截断。"},
    {"id": "estimated_grid", "evidence": "tools/diff-data.py:53-79; tools/diff-data32.py:154-181",
     "finding": "前两版以相邻行间隔中位数猜网格，未解析 TimingPoints，不能当真实 1/16 或 1/32 拍点。"},
    {"id": "row_rate_not_nps", "evidence": "tools/diff-data.py:68-70; tools/diff-data32.py:166-167; tools/diff-data-true.py:81-82",
     "finding": "len(rows)/秒是 RPS（同行和弦合并后行率），不是按各键起点计数的 NPS。密度/占位率桶也不是经过校准的真实难度。"},
    {"id": "ln_duration_lost", "evidence": "tools/diff-data.py:16-43; tools/diff-data32.py:57-84; tools/diff-data-true.py:34-49,86-91",
     "finding": "type&128 仅留下 LN 起点，结束时间及 hold/release 均未保存；旧二值卷无法恢复 LN 时值。"},
    {"id": "partial_provenance_not_a_certificate", "evidence": "tools/build-cond-index.py:326-360; out/diff-data/cond-index.json:2-6",
     "finding": "cond-index 是6首音频的内容重放候选索引；歧义时仍取 picks[0]，不是全库唯一 source_id。unresolved_provenance=0 不代表每行得到唯一曲目身份。"},
]

WINDOW_INTERFACE = {
    "source_id": "每个窗口的 chart_id（本 manifest 的内容寻址 ID），Unicode/string [N]；不要用行号替代。",
    "group_id": "每个窗口继承 chart.group_id，同曲/同源/重复连通分量的 ID，string [N]。",
    "split": "每个窗口原样继承 chart.split，train/val/test；quarantine 和 rejected 不切窗。禁止再做窗口随机切分。",
    "start_cell": "int64 [N]，在该 chart 固定 timing 原点上的绝对起始格，不得每窗把首音符归零。",
    "window_cells": "int，窗口长；区间 [start_cell,start_cell+window_cells)，同时保存 grid_division=8（每拍8格=1/32）。",
    "timing": "chart.timing_segments 保存红线 offset_ms/beat_length_ms/beat_start/meter；cell(t)=round(8*beat_at_ms(t))，支持变速。",
    "ln": "至少保存 onset/hold/release 三通道，或另存 lane,start_ms,end_ms；跨窗 LN 必须有 carry-in/carry-out，不可只截起点。",
    "density": "分别叫 row_rate_rps、note_head_rate_nps、occupancy；密度条件不能标注为真实难度。归一化参数只能在 train 拟合。",
    "audit": "保存 manifest_sha256、原始 file_sha256、content_sha256、parser/grid 版本；按 group_id/source_id/content hash 验证集间无交集，按 source_id 和半开格区间检查重叠窗。旧 NPZ 需重新从原谱构建来源，不能按相似窗口猜唯一来源。",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def normalise(value):
    return "".join(c for c in unicodedata.normalize("NFKC", value).casefold() if c.isalnum())


def number(value):
    v = float(value)
    if not math.isfinite(v):
        raise ValueError("non-finite number")
    return int(v) if v.is_integer() else v


def beat_at_ms(segments, time_ms):
    """Continuous beat coordinate, anchored at the first red timing offset.

    Negative cells before the first red point are allowed; green points never
    enter segments. This is a beat coordinate, not an assumed fixed 4/4 bar.
    """
    i = max(0, bisect.bisect_right([s["offset_ms"] for s in segments], time_ms) - 1)
    s = segments[i]
    return s["beat_start"] + (time_ms - s["offset_ms"]) / s["beat_length_ms"]


def parse_chart(path):
    """Parse only the appropriate .osu sections; preserve LN end times.

    Return both errors and parsed evidence. Any malformed playable/timing row
    rejects the entire chart, rather than silently training on a partial parse.
    """
    raw = Path(path).read_bytes()
    result = {"file_sha256": hashlib.sha256(raw).hexdigest(), "errors": [],
              "warnings": [], "notes": [], "timing_segments": [], "metadata": {},
              "legacy_timing_candidate": None, "legacy_effects_one_lines": 0}
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        result["errors"].append("invalid_utf8")
        return result
    headers = defaultdict(dict)
    red = []
    section = ""
    for lineno, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        if s.startswith("[") and s.endswith("]"):
            section = s[1:-1].casefold()
            continue
        fields = s.split(",")
        # Reproduce the *read-only* legacy timing predicate for audit evidence.
        if len(fields) >= 3 and fields[-1].strip() == "1":
            result["legacy_effects_one_lines"] += 1
        if result["legacy_timing_candidate"] is None and len(fields) >= 2:
            try:
                bl = number(fields[1])
                un = int(fields[6]) if len(fields) > 6 else 1
                if un == 1 and bl > 0:
                    result["legacy_timing_candidate"] = {"section": section, "line": lineno,
                                                          "beat_length_ms": bl}
            except (ValueError, OverflowError):
                pass
        if section in ("general", "metadata", "difficulty") and ":" in s:
            key, value = s.split(":", 1)
            key, value = key.strip().casefold(), value.strip()
            if key in headers[section] and headers[section][key] != value:
                result["errors"].append(f"conflicting_header:{section}.{key}:{lineno}")
            headers[section][key] = value
        elif section == "timingpoints":
            try:
                offset, bl = number(fields[0]), number(fields[1])
                un = int(fields[6]) if len(fields) > 6 else 1
                if un not in (0, 1):
                    raise ValueError("invalid uninherited flag")
                if un == 1:
                    meter = int(fields[2]) if len(fields) > 2 else 4
                    if bl <= 0 or meter <= 0:
                        raise ValueError("nonpositive red beat length or meter")
                    red.append((offset, bl, meter))
            except (ValueError, IndexError, OverflowError):
                result["errors"].append(f"invalid_timing_point:{lineno}")
        elif section == "hitobjects":
            try:
                x, start, typ = number(fields[0]), number(fields[2]), int(fields[3])
                if not (0 <= x < 512) or len(fields) < 5:
                    raise ValueError("invalid hit object")
                lane = int(x * 4 // 512)
                if typ & 128:
                    end = number(fields[5].split(":", 1)[0])
                    if end <= start:
                        raise ValueError("nonpositive LN duration")
                elif typ & 1:
                    end = start
                else:
                    raise ValueError("unsupported mania object type")
                result["notes"].append((lane, start, end))
            except (ValueError, IndexError, OverflowError):
                result["errors"].append(f"invalid_hit_object:{lineno}")
    result["metadata"] = headers["metadata"]
    result["audio_filename"] = headers["general"].get("audiofilename", "")
    for section_name, key, expected, code in (("general", "mode", 3, "mode_not_3"),
                                              ("difficulty", "circlesize", 4, "keys_not_4")):
        try:
            value = number(headers[section_name].get(key, ""))
        except ValueError:
            value = None
        result["mode" if key == "mode" else "keys"] = value
        if value != expected:
            result["errors"].append(code)
    previous = None
    for offset, bl, meter in sorted(set(red)):
        if previous and offset == previous["offset_ms"]:
            result["errors"].append(f"conflicting_red_timing:{offset}")
            continue
        beat = 0 if previous is None else previous["beat_start"] + (offset - previous["offset_ms"]) / previous["beat_length_ms"]
        previous = {"offset_ms": offset, "beat_length_ms": bl, "meter": meter, "beat_start": beat}
        result["timing_segments"].append(previous)
    if not red:
        result["errors"].append("missing_red_timing")
    if not result["notes"]:
        result["errors"].append("no_notes")
    result["notes"].sort(key=lambda n: (n[1], n[0], n[2]))
    if not result["errors"]:
        # Exclude labels, filenames, hit sounds, and formatting from gameplay
        # deduplication, but never discard timing offsets or LN duration.
        result["content_sha256"] = digest({"format": "gj-4k-gameplay-v1", "mode": 3, "keys": 4,
                                           "red_timing": sorted(set(red)), "notes": result["notes"]})
    return result


def identity_info(parsed, path, pack_dir, audio_cache):
    meta = parsed["metadata"]
    keys, missing = set(), []
    try:
        set_id = int(meta.get("beatmapsetid", ""))
        if set_id <= 0:
            raise ValueError("nonpositive ID")
        keys.add(f"set:{set_id}")
    except ValueError:
        set_id = None
        missing.append("positive_BeatmapSetID")
    titles = {normalise(meta.get(k, "")) for k in ("title", "titleunicode")} - UNKNOWN
    artists = {normalise(meta.get(k, "")) for k in ("artist", "artistunicode")} - UNKNOWN
    if not titles:
        missing.append("Title_or_TitleUnicode")
    if not artists:
        missing.append("Artist_or_ArtistUnicode")
    for artist in artists:
        for title in titles:
            keys.add(f"meta:{artist}|{title}")
    filename = parsed["audio_filename"].replace("\\", "/").strip('"')
    audio_hash = None
    if not filename:
        missing.append("AudioFilename")
    else:
        candidate = (path.parent / filename).resolve()
        if not candidate.is_relative_to(pack_dir.resolve()):
            missing.append("unsafe_or_external_audio_path")
        elif candidate.is_file():
            if candidate not in audio_cache:
                h = hashlib.sha256()
                with candidate.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        h.update(block)
                audio_cache[candidate] = h.hexdigest()
            audio_hash = audio_cache[candidate]
            keys.add(f"audio_sha256:{audio_hash}")
        else:
            missing.append("audio_file_unavailable")
        # Conservative grouping by informative referenced basename also works
        # with .osu-only packs. It is NOT a verified audio fingerprint.
        audio_name = normalise(Path(filename).stem)
        if audio_name not in GENERIC_AUDIO and len(audio_name) >= 4:
            keys.add(f"audio_name:{audio_name}")
    return {"keys": sorted(keys), "missing": missing, "beatmap_set_id": set_id,
            "audio_sha256": audio_hash, "audio_filename": filename}


def npz_header_audit(root):
    """Inspect only ZIP/NPY headers, without importing numpy/torch or arrays."""
    result = []
    for folder in ("diff-data", "diff-data32", "diff-data-true"):
        path = root / "out" / folder / "dataset.npz"
        if not path.is_file():
            continue
        rec = {"path": path.relative_to(root).as_posix(), "arrays": {},
               "no_leakage_proven": False}
        try:
            with zipfile.ZipFile(path) as archive:
                for member in sorted(archive.namelist()):
                    if not member.endswith(".npy"):
                        continue
                    with archive.open(member) as stream:
                        prefix = stream.read(8)
                        if prefix[:6] != b"\x93NUMPY" or prefix[6] not in (1, 2, 3):
                            raise ValueError("unsupported NPY header")
                        nbytes = 2 if prefix[6] == 1 else 4
                        length = int.from_bytes(stream.read(nbytes), "little")
                        if length > 65536:
                            raise ValueError("oversized NPY header")
                        header = ast.literal_eval(stream.read(length).decode("utf-8" if prefix[6] == 3 else "latin1").strip())
                    rec["arrays"][member[:-4]] = {"shape": list(header["shape"]), "dtype": header["descr"]}
            required = {"source_id", "group_id", "start_cell", "split"}
            rec["missing_provenance_fields"] = sorted(required - rec["arrays"].keys())
            rec["status"] = "cannot_prove_no_leakage" if rec["missing_provenance_fields"] else "requires_value_and_split_validation"
        except (OSError, ValueError, SyntaxError, KeyError, zipfile.BadZipFile) as exc:
            rec["status"] = f"unreadable:{type(exc).__name__}"
        result.append(rec)
    provenance = root / "out/diff-data/cond-index.json"
    if provenance.is_file():
        try:
            data = json.loads(provenance.read_text(encoding="utf-8"))
            result.append({"path": provenance.relative_to(root).as_posix(),
                           **{k: data.get(k) for k in ("dataset_windows", "matched_windows", "unresolved_provenance", "ambiguous_multi_source")},
                           "no_leakage_proven": False, "status": "partial_audio_index_not_full_source_certificate"})
        except (OSError, ValueError) as exc:
            result.append({"path": provenance.relative_to(root).as_posix(), "status": str(exc), "no_leakage_proven": False})
    return result


def split_groups(groups, seed, val_fraction, test_fraction):
    if not (0 <= val_fraction < 1 and 0 <= test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("val/test fractions must be nonnegative with sum < 1")
    eligible = sorted((g for g in groups if g["identity_keys"]), key=lambda g: g["group_id"])
    random.Random(seed).shuffle(eligible)
    n = len(eligible)
    nv, nt = int(n * val_fraction), int(n * test_fraction)
    if n >= 3:
        nv = max(nv, int(val_fraction > 0))
        nt = max(nt, int(test_fraction > 0))
        while nv + nt >= n:
            if nv >= nt:
                nv -= 1
            else:
                nt -= 1
    for i, group in enumerate(eligible):
        group["split"] = "val" if i < nv else "test" if i < nv + nt else "train"
    for group in groups:
        group.setdefault("split", "quarantine")


def leakage_assertions(charts, groups):
    fields = ("group_id", "chart_id", "content_sha256", "file_sha256", "identity_key")
    sets = {s: {f: set() for f in fields} for s in SPLITS}
    for chart in charts:
        if chart["split"] not in sets:
            continue
        dst = sets[chart["split"]]
        for field in ("group_id", "chart_id", "content_sha256"):
            dst[field].add(chart[field])
        dst["file_sha256"].update(a["file_sha256"] for a in chart["occurrences"])
        dst["identity_key"].update(chart["identity_keys"])
    pairs = {}
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            pairs[f"{left}/{right}"] = {f: len(sets[left][f] & sets[right][f]) for f in fields}
    result = {"pairwise_intersections": pairs, "all_pairwise_intersections_zero": all(n == 0 for p in pairs.values() for n in p.values()),
              "every_chart_in_exactly_one_group": sorted(c["chart_id"] for c in charts) == sorted(c for g in groups for c in g["chart_ids"]),
              "all_windows_must_inherit_group_split": True,
              "window_overlap_audited": False,
              "existing_npz_no_leakage_proven": False,
              "scope": "observed BeatmapSetID/metadata/audio-reference/audio-bytes/gameplay-hash identities; not an oracle for unlabelled or edited recordings"}
    assert result["all_pairwise_intersections_zero"]
    assert result["every_chart_in_exactly_one_group"]
    return result


def build_manifest(root, pack_dir="out/packs", seed=136, val_fraction=0.1, test_fraction=0.1,
                   audit_existing=True):
    root = Path(root).resolve()
    pack_dir = (root / pack_dir).resolve()
    if not pack_dir.is_relative_to(root):
        raise ValueError("pack directory must be within root")
    files = [(style, path) for style in STYLES for path in sorted((pack_dir / style).rglob("*.osu"))]
    if not files:
        raise ValueError("no .osu charts in the four style directories")
    by_content, rejected, fingerprints, audio_cache = {}, [], [], {}
    observed = Counter()
    missing_counts = Counter()
    for style, path in files:
        rel = path.relative_to(root).as_posix()
        try:
            parsed = parse_chart(path)
        except OSError as exc:
            rejected.append({"path": rel, "style": style, "errors": [f"read_error:{type(exc).__name__}"]})
            fingerprints.append([rel, "unreadable"])
            continue
        fingerprints.append([rel, parsed["file_sha256"]])
        candidate = parsed["legacy_timing_candidate"]
        if candidate and candidate["section"] != "timingpoints":
            observed["legacy_timing_first_candidate_outside_timingpoints"] += 1
        if parsed["legacy_effects_one_lines"] != len(parsed["timing_segments"]):
            observed["legacy_red_count_differs_from_actual_red_count"] += 1
        if parsed["legacy_effects_one_lines"] > 3:
            observed["legacy_variable_bpm_skip_predicate"] += 1
        if parsed["errors"]:
            rejected.append({"path": rel, "style": style, "file_sha256": parsed["file_sha256"],
                             "errors": parsed["errors"]})
            continue
        identity = identity_info(parsed, path, pack_dir, audio_cache)
        missing_counts.update(identity["missing"])
        notes = parsed["notes"]
        rows = len({n[1] for n in notes})
        ln_count = sum(end > start for _, start, end in notes)
        segments = parsed["timing_segments"]
        observed["accepted_note_heads"] += len(notes)
        observed["accepted_distinct_onset_rows"] += rows
        observed["accepted_ln_heads"] += ln_count
        observed["accepted_charts_with_ln"] += int(ln_count > 0)
        variable = len({s["beat_length_ms"] for s in segments}) > 1
        observed["accepted_charts_with_tempo_changes"] += int(variable)
        duration = (max(n[2] for n in notes) - min(n[1] for n in notes)) / 1000
        rates = {"row_rate_rps": round(rows / duration, 6) if duration > 0 else None,
                 "note_head_rate_nps": round(len(notes) / duration, 6) if duration > 0 else None,
                 "duration_seconds_including_ln_tails": duration,
                 "note_heads": len(notes), "distinct_onset_rows": rows, "ln_heads": ln_count}
        occurrence = {"path": rel, "style": style, "file_sha256": parsed["file_sha256"],
                      "metadata": parsed["metadata"], "missing": identity["missing"],
                      "beatmap_set_id": identity["beatmap_set_id"], "audio_filename": identity["audio_filename"],
                      "audio_sha256": identity["audio_sha256"]}
        h = parsed["content_sha256"]
        if h not in by_content:
            by_content[h] = {"chart_id": "gjc_" + h, "content_sha256": h,
                             "canonical_path": rel, "styles": [], "identity_keys": [], "occurrences": [],
                             "timing_segments": segments, "tempo_changes": variable, "statistics": rates}
        chart = by_content[h]
        chart["styles"] = sorted(set(chart["styles"]) | {style}, key=STYLES.index)
        chart["identity_keys"] = sorted(set(chart["identity_keys"]) | set(identity["keys"]))
        chart["occurrences"].append(occurrence)
    charts = sorted(by_content.values(), key=lambda c: c["chart_id"])
    parent = list(range(len(charts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner = {}
    for i, chart in enumerate(charts):
        for key in chart["identity_keys"]:
            if key in owner:
                parent[find(i)] = find(owner[key])
            else:
                owner[key] = i
    components = defaultdict(list)
    for i, chart in enumerate(charts):
        components[find(i)].append(chart)
    groups, conflicts = [], []
    for members in components.values():
        chart_ids = sorted(c["chart_id"] for c in members)
        identities = sorted({k for c in members for k in c["identity_keys"]})
        gid = "gjs_" + digest({"identity_keys": identities, "chart_ids": chart_ids})
        group = {"group_id": gid, "chart_ids": chart_ids, "identity_keys": identities,
                 "styles": sorted({s for c in members for s in c["styles"]}, key=STYLES.index)}
        if not identities:
            group["quarantine_reason"] = "no_usable_song_identity"
        groups.append(group)
        for chart in members:
            chart["group_id"] = gid
            chart["primary_style"] = chart["styles"][0]
            if len(chart["styles"]) > 1:
                conflicts.append({"kind": "duplicate_content_multiple_style_labels", "chart_id": chart["chart_id"], "styles": chart["styles"]})
        # Preserve inconsistent evidence rather than trusting a single alias.
        occurrences = [a for c in members for a in c["occurrences"]]
        for key, kind in (("beatmap_set_id", "multiple_set_ids"), ("audio_filename", "multiple_audio_references"),
                          ("audio_sha256", "multiple_audio_byte_hashes")):
            values = sorted({a[key] for a in occurrences if a[key] is not None and a[key] != ""})
            if len(values) > 1:
                conflicts.append({"kind": kind, "group_id": gid, "values": values,
                                  "resolution": "conservatively_group_together_never_split"})
        meta_values = sorted({k for k in identities if k.startswith("meta:")})
        if len(meta_values) > 1:
            conflicts.append({"kind": "multiple_title_artist_identities", "group_id": gid, "values": meta_values,
                              "resolution": "conservatively_group_together_never_split"})
    groups.sort(key=lambda g: g["group_id"])
    split_groups(groups, seed, val_fraction, test_fraction)
    group_map = {g["group_id"]: g for g in groups}
    for chart in charts:
        chart["split"] = group_map[chart["group_id"]]["split"]
    assertions = leakage_assertions(charts, groups)
    accepted_files = sum(len(c["occurrences"]) for c in charts)
    raw_unique = len({a["file_sha256"] for c in charts for a in c["occurrences"]})
    all_splits = SPLITS + ("quarantine",)
    per_style = {}
    for style in STYLES:
        style_charts = [c for c in charts if style in c["styles"]]
        per_style[style] = {"discovered_files": sum(s == style for s, _ in files),
                            "accepted_files": sum(a["style"] == style for c in charts for a in c["occurrences"]),
                            "unique_content_memberships": len(style_charts),
                            "primary_style_charts": sum(c["primary_style"] == style for c in charts),
                            "split_memberships": {s: sum(c["split"] == s for c in style_charts) for s in all_splits},
                            "split_primary_charts": {s: sum(c["split"] == s and c["primary_style"] == style for c in charts) for s in all_splits}}
    summary = {"discovered_files": len(files), "accepted_files": accepted_files,
               "rejected_files": len(rejected), "raw_byte_unique_files": raw_unique,
               "raw_byte_duplicate_files": accepted_files - raw_unique,
               "unique_charts": len(charts), "deduplicated_files": accepted_files - len(charts),
               "additional_gameplay_duplicates": raw_unique - len(charts), "groups": len(groups),
               "eligible_groups": sum(g["split"] in SPLITS for g in groups),
               "quarantined_charts": sum(c["split"] == "quarantine" for c in charts),
               "per_split": {s: {"charts": sum(c["split"] == s for c in charts),
                                 "groups": sum(g["split"] == s for g in groups),
                                 "original_files": sum(len(c["occurrences"]) for c in charts if c["split"] == s)} for s in all_splits},
               "per_style": per_style, "missing_information_file_counts": dict(sorted(missing_counts.items())),
               "conflict_records": len(conflicts), "empirical_legacy_audit": dict(sorted(observed.items()))}
    return {"schema": "gj-source-manifest-v1", "parser": "sectioned-osu-mania-4k-ln-red-timing-v1",
            "seed": seed, "split_policy": {"unit": "transitive_song_source_group", "val_fraction": val_fraction,
                                           "test_fraction": test_fraction, "fraction_basis": "group_count_not_chart_count",
                                           "rules": ["same positive BeatmapSetID", "normalised title+artist including Unicode aliases",
                                                     "same verified audio SHA256", "same informative audio reference basename (conservative, unverified)",
                                                     "same gameplay SHA256 including red timing and LN ends"],
                                           "unknown_identity": "quarantine_not_train_val_test"},
            "input_pack_dir": pack_dir.relative_to(root).as_posix(), "input_tree_sha256": digest(fingerprints),
            "summary": summary, "assertions": assertions, "charts": charts, "groups": groups,
            "rejected": rejected, "conflicts": conflicts, "legacy_code_findings": AUDIT_FINDINGS,
            "existing_npz_provenance": npz_header_audit(root) if audit_existing else [],
            "window_source_interface": WINDOW_INTERFACE,
            "limitations": ["此清单尚未接入任何训练器；旧窗口随机切分的验证数值不能据此升级为无泄漏指标。",
                            "分组保证覆盖已观测身份的传递闭包，不证明未标识、改名、转码或缺音频文件实际不是同曲；元数据无法提供音频指纹真值。",
                            "练习包共用 set ID/标题时整包保守合组，可能很大，图数/风格比例不保证80/10/10；组不能拆开补比例。",
                            "同内容多style保留全部标签并记录冲突；primary_style仅为可加总统计的固定顺序标签，不是真值裁决。",
                            "本工具只生成manifest和报告，不生成窗口/谱包、不修改原始.osu/旧NPZ、不训练或评估模型。"]}


def render_report(manifest):
    s = manifest["summary"]
    lines = ["# Round136 · GJ 数据可信划分与验收", "",
             "## 实际清点与去重", "",
             f"- 原始 .osu：{s['discovered_files']}；接收文件：{s['accepted_files']}；拒收：{s['rejected_files']}。",
             f"- 字节级重复：{s['raw_byte_duplicate_files']}；额外玩法内容重复：{s['additional_gameplay_duplicates']}；共去重：{s['deduplicated_files']}；唯一谱图：{s['unique_charts']}。",
             f"- 同源分组：{s['groups']}（可划分 {s['eligible_groups']}）；身份不明隔离谱：{s['quarantined_charts']}。",
             f"- 固定 seed={manifest['seed']}，按组数分 train/val/test，val={manifest['split_policy']['val_fraction']:.1%}、test={manifest['split_policy']['test_fraction']:.1%}、其余train；不拆大组凑图数。", 
             "- 玩法SHA256包括红线时间/拍长/拍号、每键起止时刻（含LN尾），忽略标签、文件名、音效和排版；所有原文件路径及字节SHA256仍保留。", "",
             "| 划分 | 唯一谱图 | 同源组 | 原文件（含副本） |", "|---|---:|---:|---:|"]
    for split, counts in s["per_split"].items():
        lines.append(f"| {split} | {counts['charts']} | {counts['groups']} | {counts['original_files']} |")
    lines += ["", "### 风格计数", "", "多style重复谱同时计入各style membership，因此下面四行不能直接相加；manifest另给可加总的 primary_style 固定顺序统计，不擅自裁决标签冲突。", "",
              "| style | 原始文件 | 接收文件 | 唯一内容membership | train | val | test | 隔离 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for style, c in s["per_style"].items():
        n = c["split_memberships"]
        lines.append(f"| {style} | {c['discovered_files']} | {c['accepted_files']} | {c['unique_content_memberships']} | {n['train']} | {n['val']} | {n['test']} | {n['quarantine']} |")
    lines += ["", "## 无交集断言", "", "在已观测身份的传递闭包内，按同set、标题+艺人（含Unicode）、音频字节hash、非通用音频引用名、重复玩法hash合组。以下均为本次实际计算；不是未标识音频的绝对无同曲证明。", "",
              "| 对比 | group_id | chart_id | 玩法hash | 原文件hash | 身份key |", "|---|---:|---:|---:|---:|---:|"]
    for pair, c in manifest["assertions"]["pairwise_intersections"].items():
        lines.append(f"| {pair} | {c['group_id']} | {c['chart_id']} | {c['content_sha256']} | {c['file_sha256']} | {c['identity_key']} |")
    lines += ["", f"- 所有交集为零：{manifest['assertions']['all_pairwise_intersections_zero']}。每张谱恰好属于一组：{manifest['assertions']['every_chart_in_exactly_one_group']}。",
              "- 下游先固定组划分再切窗，所有同图窗口继承同一split；不能再调用窗口randperm划分。", "",
              "## 实证问题与旧NPZ验收边界", ""]
    for f in AUDIT_FINDINGS:
        lines.append(f"- **{f['id']}**：{f['finding']} 证据：`{f['evidence']}`。")
    lines += ["", "本次原谱实数（重复文件也计数，避免与唯一谱统计混淆）：", ""]
    for k, v in s["empirical_legacy_audit"].items():
        lines.append(f"- `{k}` = {v}")
    lines += ["", "旧NPZ仅以标准库读取ZIP/NPY头（不加载torch、不训练），不是来源重建：", ""]
    for item in manifest["existing_npz_provenance"]:
        arrays = "; ".join(f"{k}:{v['shape']} {v['dtype']}" for k, v in item.get("arrays", {}).items())
        lines.append(f"- `{item['path']}`：`{item['status']}`；{arrays or json.dumps({k: v for k, v in item.items() if k not in ('path', 'status')}, ensure_ascii=False)}。")
        if item.get("missing_provenance_fields"):
            lines.append("  缺少 " + ", ".join(f"`{f}`" for f in item["missing_provenance_fields"]) + "，**不能证明旧NPZ无同曲/重叠窗泄漏**。")
    lines += ["", "## 冲突、缺失与拒收", "",
              f"冲突记录 {s['conflict_records']} 条；详见manifest.conflicts，绝不静默选择其中一个身份。多set/多音频/多标题并不自动判错（练习包可能合法），统一保守合组。", ""]
    for k, v in s["missing_information_file_counts"].items():
        lines.append(f"- 缺失 `{k}`：{v} 个接收原文件。")
    for rec in manifest["rejected"]:
        lines.append(f"- 拒收 `{rec['path']}`：{', '.join(rec['errors'])}。")
    lines += ["", "## 给后续数据构建器的最小来源接口", ""]
    for key, text in WINDOW_INTERFACE.items():
        lines.append(f"- `{key}`：{text}")
    lines += ["", "## 限制与复现", ""]
    lines.extend("- " + text for text in manifest["limitations"])
    lines += ["- 运行 `python3 tools/gj-data-manifest.py --out-dir <新的空目录>` 可复用；默认种子136。输出文件存在即拒绝覆盖。",
              "- 快测试：`python3 -m unittest discover -s tests -p 'test_gj_data_manifest.py' -v`，全部为临时目录内合成.osu，CPU/标准库，无网络。",
              f"- 输入清单指纹：`{manifest['input_tree_sha256']}`。", ""]
    return "\n".join(lines)


def write_outputs(manifest, out_dir):
    out_dir = Path(out_dir)
    paths = (out_dir / "manifest.json", out_dir / "REPORT.md")
    if any(p.exists() for p in paths):
        raise FileExistsError("refusing to overwrite existing manifest.json or REPORT.md; choose a new --out-dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    content = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with paths[0].open("x", encoding="utf-8") as stream:
        stream.write(content)
    with paths[1].open("x", encoding="utf-8") as stream:
        stream.write(render_report(manifest))
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--pack-dir", default="out/packs")
    ap.add_argument("--out-dir", default="out/diffusion/round136-data")
    ap.add_argument("--seed", type=int, default=136)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--test-fraction", type=float, default=0.1)
    args = ap.parse_args(argv)
    try:
        manifest = build_manifest(args.root, args.pack_dir, args.seed, args.val_fraction, args.test_fraction)
        paths = write_outputs(manifest, args.root.resolve() / args.out_dir)
    except (OSError, ValueError) as exc:
        ap.exit(2, f"error: {exc}\n")
    s = manifest["summary"]
    print(json.dumps({"files": s["discovered_files"], "unique_charts": s["unique_charts"],
                      "deduplicated": s["deduplicated_files"], "groups": s["groups"], "splits": s["per_split"],
                      "assertions_passed": manifest["assertions"]["all_pairwise_intersections_zero"],
                      "outputs": [str(p) for p in paths]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
