#include "MinaCalc/MinaCalc.h"
#include <vector>

extern "C" {
// times: 秒为单位（行时间），masks: 每行的按键位掩码
int mc_calc(const double* times, const unsigned int* masks, int n,
            double rate, double goal, float* out) {
    std::vector<NoteInfo> v;
    v.reserve((size_t)n);
    for (int i = 0; i < n; i++) {
        NoteInfo ni;
        ni.notes = masks[i];
        ni.rowTime = (float)times[i];
        v.push_back(ni);
    }
    Calc* calc = new Calc;
    std::vector<float> ssr = MinaSDCalc(v, (float)rate, (float)goal, calc);
    int m = (int)ssr.size();
    if (m > 8) m = 8;
    for (int i = 0; i < m; i++) out[i] = ssr[i];
    delete calc;
    return m;
}
}
