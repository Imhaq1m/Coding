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

// === DEBUG MACROS ===
#define debug(x) cerr << #x << " : " << x << endl;
#define debug2(x, y)                                                           \
  cerr << #x << " : " << x << " | " << #y << " : " << y << endl;
#define debug3(x, y, z)                                                        \
  cerr << #x << " : " << x << " | " << #y << " : " << y << " | " << #z         \
       << " : " << z << endl;

// === UTILITIES ===
template <typename T>
void print_vector(const vector<T> &v, const string &name = "") {
  if (!name.empty())
    cerr << name << " : ";
  for (const auto &e : v)
    cerr << e << " ";
  cerr << endl;
}
template <typename T> void print_set(const set<T> &s, const string &name = "") {
  if (!name.empty())
    cerr << name << " : ";
  for (const auto &e : s)
    cerr << e << " ";
  cerr << endl;
}
template <typename T, typename U>
void print_map(const map<T, U> &m, const string &name = "") {
  if (!name.empty())
    cerr << name << " : ";
  for (const auto &e : m)
    cerr << "{" << e.first << ":" << e.second << "} ";
  cerr << endl;
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

// --- BFS (Unweighted Graph) ---
vector<int> bfs(const vector<vector<int>> &adj, int start, int n) {
  vector<int> dist(n, -1);
  queue<int> q;
  dist[start] = 0;
  q.push(start);

  while (!q.empty()) {
    int u = q.front();
    q.pop();
    for (int v : adj[u]) {
      if (dist[v] == -1) {
        dist[v] = dist[u] + 1;
        q.push(v);
      }
    }
  }
  return dist;
}

// --- DFS (Recursive) ---
void dfs_recursive(const vector<vector<int>> &adj, vector<bool> &visited,
                   int u) {
  visited[u] = true;
  // Process node u here
  for (int v : adj[u]) {
    if (!visited[v]) {
      dfs_recursive(adj, visited, v);
    }
  }
}

// --- DFS (Iterative) ---
void dfs_iterative(const vector<vector<int>> &adj, int start, int n) {
  vector<bool> visited(n, false);
  stack<int> st;
  st.push(start);

  while (!st.empty()) {
    int u = st.top();
    st.pop();
    if (visited[u])
      continue;
    visited[u] = true;
    // Process node u here
    for (int v : adj[u]) {
      if (!visited[v]) {
        st.push(v);
      }
    }
  }
}

vector<int> build_suffix_array(string s) {
  int n = s.size();
  vector<int> sa(n), rank(n), temp(n);
  for (int i = 0; i < n; ++i) {
    sa[i] = i;
    rank[i] = s[i];
  }

  for (int k = 1; k < n; k <<= 1) {
    auto cmp = [&](int a, int b) {
      if (rank[a] != rank[b])
        return rank[a] < rank[b];
      int ra = (a + k < n ? rank[a + k] : -1);
      int rb = (b + k < n ? rank[b + k] : -1);
      return ra < rb;
    };
    sort(sa.begin(), sa.end(), cmp);

    temp[sa[0]] = 0;
    for (int i = 1; i < n; ++i)
      temp[sa[i]] = temp[sa[i - 1]] + cmp(sa[i - 1], sa[i]);
    rank = temp;
  }
  return sa;
}

vector<int> build_lcp_array(string s, vector<int> sa) {
  int n = s.size();
  vector<int> rank(n), lcp(n);
  for (int i = 0; i < n; ++i)
    rank[sa[i]] = i;

  int h = 0;
  for (int i = 0; i < n; ++i) {
    if (rank[i] > 0) {
      int j = sa[rank[i] - 1];
      while (i + h < n && j + h < n && s[i + h] == s[j + h])
        h++;
      lcp[rank[i]] = h;
      if (h > 0)
        h--;
    }
  }
  return lcp;
}

// --- Connected Components in Undirected Graph ---
int count_components(int n, const vector<vector<int>> &adj) {
  vector<bool> visited(n, false);
  int components = 0;

  for (int i = 0; i < n; ++i) {
    if (!visited[i]) {
      components++;
      dfs_recursive(adj, visited, i); // or use iterative version
    }
  }
  return components;
}

// --- BFS for Grid Traversal (e.g., Maze) ---
int bfs_grid(const vector<vector<char>> &grid, pii start, pii end) {
  int n = grid.size(), m = grid[0].size();
  vector<vector<int>> dist(n, vector<int>(m, -1));
  queue<pii> q;
  dist[start.fi][start.se] = 0;
  q.push(start);

  int dx[] = {-1, 0, 1, 0};
  int dy[] = {0, 1, 0, -1};

  while (!q.empty()) {
    auto [x, y] = q.front();
    q.pop();
    for (int dir = 0; dir < 4; ++dir) {
      int nx = x + dx[dir];
      int ny = y + dy[dir];
      if (nx >= 0 && nx < n && ny >= 0 && ny < m && grid[nx][ny] != '#' &&
          dist[nx][ny] == -1) {
        dist[nx][ny] = dist[x][y] + 1;
        q.push({nx, ny});
        if (make_pair(nx, ny) == end) {
          return dist[nx][ny]; // shortest path length
        }
      }
    }
  }
  return -1; // unreachable
}

// === MAIN FUNCTION ===
void solve();

ll calc(ll f, ll s) {
  ll sum = 0;
  if (f > s)
    sum++;
  if (f == s)
    sum += 0;
  if (f < s)
    sum--;
  return sum;
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

void solve() {
  ll n, m;
  cin >> n >> m;
  vector<ll> a(n);
  vector<ll> b(m);
  vector<pll> idxarr;
  vector<ll> c(m);
  rep(i, 0, n) cin >> a[i];
  rep(i, 0, m) cin >> b[i];
  rep(i, 0, m) { idxarr.push_back({b[i], i}); }
  sort(a.begin(), a.end());
  sort(idxarr.begin(), idxarr.end());
  vector<ll> ans(m);
  ll idxb = 0;
  ll idxa = 0;
  ll curr = 0;
  while (idxb < m) {
    if (idxarr[idxb].first < a[idxa] || idxa == n) {
      c[idxarr[idxb].second] = curr;
      idxb++;
    } else {
      curr++;
      idxa++;
    }
  }
  rep(i, 0, m) cout << c[i] << " ";
  cout << endl;
}
