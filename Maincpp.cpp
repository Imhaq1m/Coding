#include <bits/stdc++.h>
using namespace std;

#define ll long long
#define MOD 1000000007
#define endl "\n"
const ll INF = 1000000001;
vector<pair<ll, ll>> arr;

void solve()
{
    ll t;
    cin >> t;

    while (t--)
    {
        ll n;
        cin >> n;
        string num = to_string(n);
        string c = "";
        int count = 0;
        for (int i = num.size()-1; i>=0; --i)
        {
            c += num[i];
            count++;
            if (count==3 && i!=0)
            {
                c += ',';
                count=0;
            }
        }
        reverse(c.begin(), c.end());
        cout << c << endl;

        stringstream res;
        res << fixed << setprecision(2);

        if(n>=1000000000)
        {
            double temp = floor((n / 1000000000) * 100) / 100.0;
            if (n%1000000000==0)
                res<<fixed<<setprecision(0);
            res<<temp<< " billion";
        }
        else if(n >= 1000000)
        {
            double temp = floor((n / 1000000) * 100) / 100.0;
            if (n%1000000==0)
                res<<fixed<<setprecision(0);
            res<<temp<<" million";
        }
        else if(n>=1000)
        {
            double temp = floor((n / 1000) * 100) / 100.0;
            if (n%1000==0)
                res << fixed << setprecision(0);
            res<<temp << " thousand";
        }
        else res<<n;
        cout << res.str() << endl;
    }
}

int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    ll t = 1;
    for (int i = 0; i < t; i++)
    {
        solve();
    }
    return 0;
}
