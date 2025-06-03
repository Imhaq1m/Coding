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
  ll p, q;
  cin >> p >> q;
  ll midy = floor(q / 2);
  cout << "Midy: " << midy << endl;
  ll midx = (-1 * (p * (midy - q)));
  cout << "Midx: " << midx << endl;
  ll ans = midx * ((2 * midy) + (midx - 1)) / 2;
  cout << "ans: " << ans << endl;
  ll c = midx + 1;
  cout << "c: " << c << endl;
  // rep(i, 0, midx) {
  // ans += (midx * ((2 * midy) + (midx - 1)) / 2) - c;
  // c += midx + 1;
  //}
  ans = 0;
  vector<ll> v(midx + 1);

  rep(i, 0, midx + 2) {
    ans += i;
    v[i] = ans;
    // cout << "ans: " << ans << endl;
  }
  ans = v[midx] * (midy + 1) + v[midy] * (midx + 1);
  cout << midx + 1 << " " << v[midx + 1] << endl;
  cout << midy + 1 << " " << v[midy + 1] << endl;
  cout << "ans: " << ans << endl;
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
