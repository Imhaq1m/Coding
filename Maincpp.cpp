#include <bits/stdc++.h>
#include <vector>
using namespace std;

// === TYPEDEFS AND CONSTANTS ===
typedef long long ll;
typedef unsigned long long ull;
typedef long double ld;
typedef pair<int, int> pii;
typedef pair<ll, ll> pll;
typedef vector<int> vi;
typedef vector<ll> vl;
typedef vector<pii> vpii;
typedef vector<pll> vpll;
typedef map<int, int> mii;
typedef map<ll, ll> mll;

const int MOD = 1000000007;
const int MOD2 = 998244353;
const double EPS = 1e-9;
const double PI = acos(-1);
const ll INF = 1000000001;
const ll LINF = 1000000000000000001;

// === FAST I/O ===
void fast_io() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);
}

// === MACROS ===
#define pb push_back
#define mp make_pair
#define fi first
#define se second
#define all(x) (x).begin(), (x).end()
#define rall(x) (x).rbegin(), (x).rend()
#define sz(x) (int)(x).size()
#define rep(i, a, b) for (int i = a; i < b; ++i)
#define repr(i, a, b) for (int i = a; i >= b; --i)
#define getunique(v)                                                           \
  {                                                                            \
    sort(all(v));                                                              \
    v.erase(unique(all(v)), v.end());                                          \
  }

vl fact = {0, 1};

void solve() {
  ll m, n;
  cin >> m >> n;
  ll c = n;
  vl a;
  vl freq(1000);
  rep(i, 0, m) {
    ll t;
    cin >> t;
    if (find(a.begin(), a.end(), t) == a.end())
      a.pb(t);
    freq[t]++;
  }
  sort(a.rbegin(), a.rend());
  cout << "Reversed sort done" << endl;
  ll idx = 0;
  while (c > 0 && idx < m) {
    cout << "Current: " << a[idx] << " and freq: " << freq[a[idx]] << endl;
    if (c <= freq[a[idx]]) {
      c -= freq[a[idx]];
      cout << "Done Dollar" << endl;
    } else {
      c -= freq[a[idx]];
      idx++;
      cout << "New Dollar" << endl;
    }
    cout << "c: " << c << endl;
  }
  cout << "Type: " << idx + 1 << endl;
  ll ans = 1;
  rep(i, 0, idx + 1) {
    cout << "Curr ele: " << a[i] << endl;
    cout << "Fact: " << fact[freq[a[i]]] << endl;
    ans = ((ans * fact[freq[a[i]]]) + MOD) % MOD;
    cout << "Curr ans: " << ans << endl;
  }
  cout << ans << endl;
}

int main() {
  fast_io();
  int t;
  // t = 1;
  cin >> t;
  rep(i, 2, 20) { fact.pb(fact[i - 1] * i); }
  for (int i = 1; i <= t; i++) {
    solve();
  }
  return 0;
}
