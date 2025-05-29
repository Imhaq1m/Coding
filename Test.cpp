#include <bits/stdc++.h>
#include <cmath>
#include <iostream>
#include <set>
#include <vector>

using namespace std;

int main() {
  int a, b;
  cin >> a >> b;

  vector<pair<int, int>> known(b);
  for (int i = 0; i < b; ++i) {
    cin >> known[i].first >> known[i].second;
  }

  set<int> possible_c;

  // Try all possible number of chests per level from 1 to 100
  for (int c = 1; c <= 100; ++c) {
    bool valid = true;
    for (auto [x, p] : known) {
      int expected_level = ceil((double)x / c);
      if (expected_level != p) {
        valid = false;
        break;
      }
    }
    if (valid) {
      possible_c.insert(c);
    }
  }

  // Now calculate possible levels for chest a
  vector<int> possible_levels;
  for (int c : possible_c) {
    int level = ceil((double)a / c);
    if (find(possible_levels.begin(), possible_levels.end(), level) ==
        possible_levels.end())
      possible_levels.push_back(level);
  }

  if (possible_levels.size() == 1) {
    cout << possible_levels[0] << endl;
  } else {
    cout << -1 << endl;
  }

  return 0;
}
