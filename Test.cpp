#include <bits/stdc++.h>
#define forr(i, a, n) for (int i = a; i < n; i++)
#define forn(i, n) for (int i = 0; i < n; i++)
#define dfor(i, n) for (int i = n - 1; i >= 0; i--)
#define forall(it, v) for (auto it = v.begin(); it != v.end(); it++)
#define pb push_back
#define sz(a) ((int)a.size())
#define all(x) (x).begin(), (x).end()
#define rall(x) (x).rbegin(), (x).rend()
#define dbg(x) cout << #x << " = " << (x) << endl
#define vdbg(x)                                                                \
  {                                                                            \
    cout << '[';                                                               \
    for (auto i : x)                                                           \
      cout << fixed << setprecision(3) << i << " ";                            \
    cout << "]\n";                                                             \
  }
#define fr first
#define sc second
#define fsp(x) fixed << setprecision((x))

using namespace std;

typedef long long ll;
typedef pair<int, int> ii;

const int INF = 1e9;
const int MOD = 1e9 + 7;

int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);

  int n;
  cin >> n;
  vector<double> v(n), dp(n + 1, 0.0);
  dp[0] = 1.0; // base: 0 heads with probability 1

  forn(i, n) {
    cin >> v[i];
    v[i] /= 100.0; // convert percentage to probability
  }

  cout << "Initial dp: ";
  vdbg(dp);

  forn(i, n) {
    cout << "\n--- Processing Coin " << i << " (p = " << v[i] << ") ---\n";

    // We go backwards to avoid using updated values
    for (int j = i; j >= 0; j--) {
      if (dp[j] == 0)
        continue;

      double old_dp_j = dp[j];
      double head_contrib = v[i] * dp[j];
      double tail_contrib = (1.0 - v[i]) * dp[j];

      cout << "  From dp[" << j << "] = " << old_dp_j << ":\n";
      cout << "    → HEADS:  dp[" << j + 1 << "] += " << head_contrib << "\n";
      cout << "    → TAILS:  dp[" << j << "]   = " << tail_contrib
           << " (overwrite)\n";

      dp[j + 1] += head_contrib; // new state: one more head
      dp[j] = tail_contrib;      // update current: tails
    }

    cout << "dp after coin " << i << ": ";
    vdbg(dp);
  }

  double ans = 0;
  forr(i, n / 2 + 1, n + 1) { ans += dp[i]; }

  cout << "\nFinal Answer: " << fixed << setprecision(10) << ans << '\n';
  return 0;
}
