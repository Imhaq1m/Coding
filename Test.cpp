#include <bits/stdc++.h>
using namespace std;

const int MOD = 1e9 + 7;
const int MAXN = 1005;

long long comb[MAXN][MAXN]; // comb[n][k] = n choose k mod MOD

// Precompute combination table using dynamic programming
void precompute_comb() {
  for (int n = 0; n < MAXN; n++) {
    comb[n][0] = 1;
    comb[n][n] = 1;
    for (int k = 1; k < n; k++) {
      comb[n][k] = (comb[n - 1][k - 1] + comb[n - 1][k]) % MOD;
    }
  }
}

int main() {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);

  precompute_comb();

  int t;
  cin >> t;

  while (t--) {
    int n, m;
    cin >> n >> m;
    vector<int> v(n);
    for (int &x : v)
      cin >> x;

    // Step 1: Sort descending
    sort(v.rbegin(), v.rend());

    // Step 2: Store frequency of each value in top m
    map<int, int> freq_top;
    for (int i = 0; i < m; i++) {
      freq_top[v[i]]++;
    }

    // Step 3: Count how many times each value appears in full list
    map<int, int> freq_all;
    for (int x : v) {
      freq_all[x]++;
    }

    // Step 4: Multiply combinations
    long long ans = 1;
    for (auto [val, needed] : freq_top) {
      int total = freq_all[val];
      ans = ans * comb[total][needed] % MOD;
    }

    cout << ans << "\n";
  }

  return 0;
}
