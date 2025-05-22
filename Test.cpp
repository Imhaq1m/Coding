#include <bits/stdc++.h>
using namespace std;

#define ll long long
#define MOD 1000000007
#define endl "\n"

// Efficient prime checking function
bool isPrime(ll num) {
    if (num <= 1) return false;
    if (num <= 3) return true; 
    if (num % 2 == 0 || num % 3 == 0) return false;
    for (ll i = 5; i * i <= num; i += 6) {
        if (num % i == 0 || num % (i + 2) == 0) return false;
    }
    return true;
}

ll factorialMod(ll n) {
    ll res = 1;
    for (ll i = 2; i <= n; i++) {
        res = (res * i) % MOD;
    }
    return res;
}

void solve() {
    ll n;
    cin >> n;

    if (!isPrime(n)) {
        cout << "NO" << endl;
        return;
    }

    string num = to_string(n);
    string revNum = num;
    reverse(revNum.begin(), revNum.end());
    if (num != revNum) {
        cout << "NO" << endl;
        return;
    }

    cout << factorialMod(n) << endl;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    ll t = 1;
    while (t--) {
        solve();
    }

    return 0;
}