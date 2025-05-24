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

void solve() {
  ll n, m, l = 0, h = 0;
  cin >> n >> m;
  vl a(n);
  rep(i, 0, n) cin >> a[i];
  rep(i, 0, n) {
    l = max(l, a[i]);
    h += a[i];
  }

  ll ans = h;
  while (l <= h) {
    ll mid = (l + h) / 2;
    ll c = 1, curr = 0;
    bool can = false;
    rep(i, 0, (ll)a.size()) {
      if (a[i] > mid) {
        can = false;
        break;
      } else if (curr + a[i] > mid) {
        c++;
        curr = a[i];
        if (c > m) {
          can = false;
          break;
        }
      } else {
        curr += a[i];
      }
    }
    if (c <= m)
      can = true;
    else
      can = false;
    if (can) {
      ans = mid;
      h = mid - 1;
    } else {
      l = mid + 1;
    }
  }
  cout << ans << endl;
}

int main() {
  fast_io();
  int t;
  t = 1;
  // cin >> t;
  for (int i = 1; i <= t; i++) {
    solve();
  }
  return 0;
}
