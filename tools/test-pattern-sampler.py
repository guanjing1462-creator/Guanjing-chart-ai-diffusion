"""Regression tests: empirical sampler coverage, units and requested row size."""
import importlib.util
import random
from collections import Counter
import numpy as np
from pathlib import Path
spec = importlib.util.spec_from_file_location('gj', Path(__file__).with_name('paint-gj.py'))
gj = importlib.util.module_from_spec(spec); spec.loader.exec_module(gj)

def state():
    return dict(prev_col=0, prev_cols=[0], recent=[0], load=Counter({k: 1 for k in range(4)}))

def call(size, gap, ms, st):
    return gj.sample_cols(random.Random(42), st, np.ones((4,4))/4, gap,
                          None, size, {'hand_alt':0.5}, 1., pitch_w=0, cell_ms=ms)

# Same time gap at two tempos must select the same empirical bucket.
gj.PATTERN[0] = ('stream', {'types': {'stream': {
    'trans': {'[0]|8': {'[2]':1.0}, '[0]|16': {'[1]':1.0},
              '[0]|32': {'[1, 3]':1.0, '[2]':999.0}}, 'marg':{}}}})
for gap, ms in [(2,75), (1,150)]:
    st=state(); assert call(1,gap,ms,st)==[2]; assert st['pattern_hits']==1
st=state(); assert call(2,.5,81,st)==[1,3], 'must retain requested cardinality'
# A missing double context must not silently return an available singleton.
gj.PATTERN[0][1]['types']['stream']['trans']['[0]|32']={'[2]':1.0}
st=state(); assert len(call(2,.5,81,st))==2; assert st['pattern_fallbacks']==1
# Jack types may sample intended repeats; non-jack types must not bypass filters.
gj.PATTERN[0]=('jack',{'types':{'jack':{'trans':{'[0]|16':{'[0]':1.0}},'marg':{}}}})
st=state(); assert call(1,1,81,st)==[0]; assert st['pattern_hits']==1
print('PASS: singleton path, tempo-independent buckets, size preservation, jack sampling')
