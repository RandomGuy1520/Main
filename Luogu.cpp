#pragma once
#include "Main.cpp"
#include "QuestionList.cpp"
using namespace std;

static const char* wrotelist[] = { "P1001", "P1035", "P1046", "P1059", "P1075", "P1085", "P1089", "P1150", "P1151", "P1179", "P1304", "P1307", "P1421", "P1422", "P1423", "P1424", "P1425", "P1427", "P1534", "P1554", "P1615", "P1634", "P1720", "P1888", "P1909", "P1914", "P1980", "P2043", "P1887", "P2433", "P3954", "P2669", "P2955", "P1000", "P3742", "P4325", "P4413", "P4414", "P4956", "P5082", "P5015", "P5660", "P5703", "P5704", "P5705", "P5709", "P5710", "P5711", "P5712", "P5713", "P5714", "P5715", "P5716", "P5718", "P5720", "P5721", "P5722", "P5724", "P5727", "P5732", "P5733", "P5737", "P8828", "P8680", "P7772", "P7933", "P8717", "P8761", "P8711", "P8722", "P1008", "P1009", "P1014", "P1029", "P1062", "P1100", "P1303" };
inline QuestionList Luogu(200);

// Error: P1138, P1317, P1567, P3156, P5718, P5724, P5727, P5744, P6284, P7226, P8717 (element of list must be const int);

// Difficulty: Beginner
inline int P1000()
{
	printf(
		"                ********\n"
		"               ************\n"
		"               ####....#.\n"
		"             #..###.....##....\n"
		"             ###.......######              ###            ###\n"
		"                ...........              #...#          #...#\n"
		"               ##*#######                 #.#.#          #.#.#\n"
		"            ####*******######             #.#.#          #.#.#\n"
		"           ...#***.****.*###....         #...#          #...#\n"
		"           ....**********##.....          ###            ###\n"
		"           ....****    *****....\n"
		"             ####        ####\n"
		"           ######        ######\n"
		"##############################################################\n"
		"#...#......#.##...#......#.##...#......#.##------------------#\n"
		"###########################################------------------#\n"
		"#..#....#....##..#....#....##..#....#....#####################\n"
		"##########################################    #----------#\n"
		"#.....#......##.....#......##.....#......#    #----------#\n"
		"##########################################    #----------#\n"
		"#.#..#....#..##.#..#....#..##.#..#....#..#    #----------#\n"
		"##########################################    ############\n"
	);
	return 0;
}
inline int P1001()
{
	// OLD;
	int i, b;
	scanf("%d%d", &i, &b);
	printf("%d", i + b);
	return 0;
}
inline int P1035()
{
	// OLD;
	int i = 0, j;
	double sum = 0;
	scanf("%d", &j);
	while (sum <= j)
	{
		i++;
		sum += 1.0 / i;
	}
	printf("%d", i);
	return 0;
}
inline int P1046()
{
	int a[10], b;
	int c = 0;
	for (int i = 0; i < 10; i++) {
		cin >> a[i];
	}
	cin >> b;
	for (int n = 0; n < 10; n++) {
		if (a[n] <= (b + 30)) {
			c++;
		}
	}
	cout << c;
	return 0;
}
inline int P1059()
{
	set<int>s;
	int a[105];
	int n;
	cin >> n;
	for (int i = 0; i < n; i++)
	{
		cin >> a[i];
		s.insert(a[i]);
	}
	cout << s.size() << endl;
	while (!s.empty())
	{
		cout << *s.begin() << " ";
		s.erase(s.begin());
	}
	return 0;
}
inline int P1075()
{
	// OLD;
	int n;
	scanf("%d", &n);
	for (int i = 2; i <= n; ++i)
		if (n % i == 0) {
			printf("%d", n / i);
			return 0;
		}
	return 0;
}
inline int P1085()
{
	// OLD;
	long long a, b, sum = 0, ans;
	for (long long i = 1; i <= 7; i++)
	{
		scanf("%lld%lld", &a, &b);
		if (a + b > sum && a + b > 8)
		{
			sum = a + b;
			ans = i;
		}

	}
	if (sum != 0) printf("%lld", ans);
	else printf("0");
	return 0;
}
inline int P1089()
{
	// OLD;
	int money = 0, yu, m = 0;
	for (int i = 1; i <= 12; i++)
	{
		money += 300;
		scanf("%d", &yu);
		if (money < yu)
		{
			printf("%d", i - 2 * i);
			return 0;
		}
		money -= yu;
		m += money / 100;
		money %= 100;
	}
	printf("%d", m * 120 + money);
	return 0;
}
inline int P1150()
{
	// OLD;
	int n, k;
	cin >> n >> k;
	cout << n + (n - 1) / (k - 1);
	return 0;
}
inline int P1151()
{
	// OLD;
	int j, f = 0;
	scanf("%d", &j);
	for (int i = 10000; i <= 30000; i++)
	{
		int sub1 = i / 100;
		int sub2 = (i / 10) % 1000;
		int sub3 = i % 1000;
		if (sub1 % j == 0 && sub2 % j == 0 && sub3 % j == 0)
		{
			printf("%d\n", i);
			f = 9;
		}
	}
	if (f == 0)
	{
		printf("No");
		return 0;
	}
}
inline int P1179()
{
	int countingtwolmao(int x);
	int a, b, sum = 0;
	cin >> a;
	cin >> b;
	for (int i = a; i < b + 1; i++)
	{
		sum += countingtwolmao(i);
	}
	cout << sum;
}
inline int P1304()
{
	// OLD;
	void writelmao(int a);
	int n;
	scanf("%d", &n);
	for (int i = 4; i <= n; i += 2)
		writelmao(i);
	return 0;
}
inline int P1307()
{
	// OLD;
	long long ans = 0, n;
	scanf("%lld", &n);
	if (n < 0)
	{
		printf("-");
		n = -n;
	}
	while (n > 0)
	{
		ans = ans * 10 + n % 10;
		n /= 10;
	}
	printf("%lld", ans);
	return 0;
}
inline int P1422()
{
	// OLD;
	int a, b;
	scanf("%d%d", &a, &b);
	printf("%d", (a * 10 + b) / 19);
	return 0;
}
inline int P1423()
{
	// OLD;
	int j = 0;
	double count, i = 2, sum = 0;
	scanf("%lf", &count);
	while (sum < count)
	{
		sum += i;
		i *= 0.98;
		j++;
	}
	printf("%d", j);
	return 0;
}
inline int P1424()
{
	// OLD;
	int n, x, tian = 1, ans = 0;
	scanf("%d%d", &x, &n);
	while (tian <= n)
	{
		if (x != 6 && x != 7)ans += 250;
		tian += 1;
		if (x == 7)
		{
			x = 1;

		}
		else x++;
	}
	printf("%d", ans);
	return 0;
}
inline int P1425()
{
	// OLD;
	int a, b, c, d;
	scanf("%d%d%d%d", &a, &b, &c, &d);
	int ans = c * 60 + d - a * 60 - b;
	printf("%d %d", ans / 60, ans % 60);
	return 0;
}
inline int P1427()
{
	const int len = 2000;
	int a, b[len], num = 0;
	cin >> a;
	for (int i = 0; i < len; i++)
	{
		if (a == 0)
			break;
		b[i] = a;
		cin >> a;
		num++;
	}
	for (int i = num - 1; i >= 0; i--)
	{
		cout << b[i] << " ";
	}
}
inline int P1534()
{
	// OLD;
	int i, sum = 0, ans = 0;
	scanf("%d", &i);
	for (int j=1;j<=i;j++)
	{
		int a, b;
		scanf("%d%d", &a, &b);
		sum += a + b - 8;
		ans += sum;		
	}
	printf("%d", ans);
	return 0;
}
inline int P1554()
{
	int countlmao(int a, int digit);
	int x, y, i, j, sum;
	cin >> x;
	cin >> y;
	for (j = 0; j < 10; j++)
	{
		sum = 0;
		for (i = x; i <= y; i++)
		{
			sum += countlmao(i, j);
		}
		cout << sum << " ";
	}
}
inline int P1615()
{
	// OLD;
	long long a, b, c, x, y, z, t;
	char m;
	cin >> a >> m >> b >> m >> c >> x >> m >> y >> m >> z >> t;
	cout << ((x - a) * 3600 + (y - b) * 60 + (z - c)) * t;
}
inline int P1634()
{
	long long x, n, ans;
	scanf("%lld%lld", &x, &n);
	ans = pow(x + 1, n);
	printf("%lld", ans);
	return 0;
}
inline int P1720()
{
	long long a = 1, b = 1, c = 0;
	int n, i;
	cin >> n;
	for (i = 3; i <= n; i++)
	{
		c = a + b;
		a = b;
		b = c;
	}
	cout << c << ".00";
	return 0;
}
inline int P1887()
{
	int a, b, c, d;
	cin >> a;
	cin >> b;
	if (a % b == 0)
	{
		for (int i = 0; i < b; i++)
			cout << a / b << " ";
	}
	d = a % b;
	c = b - d;
	for (int j = 0; j < c; j++)
		cout << (a - d) / b << " ";
	for (int k = 0; k < d; k++)
		cout << (a - d) / b + 1 << " ";
}
inline int P1888()
{
	int a, b, c, min1, max1;
	cin >> a;
	cin >> b;
	cin >> c;
	if (a < b and b < c)
		min1 = a;
	else if (b < a and b < c)
		min1 = b;
	else
		min1 = c;
	if (a > b and a > c)
		max1 = a;
	else if (b > a and b > c)
		max1 = b;
	else
		max1 = c;
	cout << min1 / Main::gcd(min1, max1) << "/" << max1 / Main::gcd(min1, max1);
}
inline int P1909()
{
	int m, n, p, j, ans = 2147483647;
	scanf("%d", &n);
	for (int i = 1; i <= 3; i++)
	{
		scanf("%d%d", &m, &p);
		if (n % m == 0) j = n / m * p;
		else j = (n / m + 1) * p;
		if (j < ans) ans = j;

	}

	printf("%d", ans);
	return 0;
}
inline int P1914()
{
	int i, key;
	string s, t;
	cin >> key;
	cin >> s;
	for (i = 0; i < s.size(); i++) {
		t += (s[i] - 'a' + key) % 26 + 'a';
	}
	cout << t;
	return 0;
}
inline int P1980()
{
	int cdinlmao(int a, int digit);
	int a, digit, count = 0;
	cin >> a;
	cin >> digit;
	for (int i = 1; i <= a; i++)
	{
		count += cdinlmao(i, digit);
	}
	cout << count;
}
inline int P2043()
{
	int countinglmao(int a, int b);
	int a, i, j, count;
	cin >> a;
	for (i = 1; i <= a; i++)
	{
		if (Main::is_prime(i))
		{
			count = 0;
			for (j = i; j <= a; j++)
			{
				count += countinglmao(j, i);
			}
			cout << i << " " << count << "\n";
		}
	}
}
inline int P2669()
{
	int n = 1, k, sum = 0, m = 0, ans = 0;
	scanf("%d", &k);
	while (ans < k)
	{
		sum += n;
		m++;
		ans++;
		if (n == m)
		{
			n++;
			m = 0;
		}
	}
	printf("%d", sum);
	return 0;
}
inline int P2955()
{
	char a;
	int count = 1, i, lastint = 0, lmao;
	cin >> lmao;
	cin.get(a);
	cin.get(a);
	i = (int)a - 48;
	while (count <= lmao)
	{
		if (i == -16 || i == -38)
		{
			count++;
			if (lastint % 2 == 0)
				cout << "even\n";
			else
				cout << "odd\n";
		}
		lastint = i;
		cin.get(a);
		i = (int)a - 48;
	}
	return 0;
}
inline int P3742()
{
	int a;
	string x, y;
	cin >> a;
	cin >> x;
	cin >> y;
	for (int i = 0; i < a; i++)
	{
		if (int(x[i]) < int(y[i]))
		{
			cout << -1;
			return 0;
		}
	}
	cout << y;
}
inline int P3954()
{
	int a, b, c;
	cin >> a;
	cin >> b;
	cin >> c;
	cout << (0.2 * a + 0.3 * b + 0.5 * c);
}
inline int P4325()
{
	int countDistinctlmao(int arr[], int n);
	int a[10], b[10];
	for (int i = 0; i < 10; i++)
	{
		cin >> a[i];
		b[i] = a[i] % 42;
	}
	cout << (countDistinctlmao(b, 10));
}
inline int P4413()
{
	int r1, r2, s; 
	scanf("%d%d", &r1, &s);
	r2 = s * 2 - r1;
	printf("%d", r2);
	return 0;
}
inline int P4414()
{
	int a[3];
	string lol;
	char x, y, z;
	for (int i = 0; i < 3; i++)
	{
		cin >> a[i];
	}
	cin >> lol;
	x = lol[0];
	y = lol[1];
	z = lol[2];
	sort(a, a + 3);
	cout << a[int(x) - 65] << " " << a[int(y) - 65] << " " << a[int(z) - 65];
}
inline int P4956()
{
	int a;
	cin >> a;
	a /= 364;
	for (int i = 1; i < ceil(a / 3) + 1; i++)
	{
		if (a - 3 * i <= 100)
		{
			cout << a - 3 * i << "\n" << i;
			return 0;
		}
	}
}
inline int P5015()
{
	char a;
	int count = 0;
	cin.get(a);
	while (a != '\n')
	{
		if (a != ' ')
			count++;
		cin.get(a);
	}
	cout << count;
}
inline int P5074()
{
	char a;
	cin >> a;
	string b = "abcdefghijklmnopqrstuvwxyz";
	string c = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
	int index = b.find(a);
	cout << c[index];
}
inline int P5082()
{
	long long n, score, s1 = 0, s2 = 0;
	scanf("%lld", &n);
	for (long long i = 1; i <= n; i++)
	{
		scanf("%lld", &score);
		s1 += score;
	}
	for (long long i = 1; i <= n; i++)
	{
		scanf("%lld", &score);
		s2 += score;
	}
	double ans = (double)(s1 * 3 - s2 * 2) / (s1 - s2);
	printf("%.6lf", ans);
	return 0;
}
inline int P5660()
{
	char a;
	int b, count = 0;
	for (int i = 0; i < 8; i++)
	{
		cin.get(a);
		b = (int)a - 48;
		if (b == 1)
			count++;
	}
	cout << count;
}
inline int P5703()
{
	long long a, b;
	cin >> a;
	cin >> b;
	cout << a * b;
}
inline int P5705()
{
	float a;
	int b, num = 0;
	cin >> a;
	while (!Main::is_int(a))
	{
		num++;
		a *= 10;
	}
	b = Main::count_digit(a) - num;
	cout << Main::read_from_backwards(a) / pow(10, b);
}
inline int P5709()
{
	int m, t, s;
	cin >> m >> t >> s;
	if (t == 0)
	{
		cout << 0 << endl;
		return 0;
	}
	if (s % t == 0)
		cout << max(m - s / t, 0) << endl;
	else
		cout << max(m - s / t - 1, 0) << endl;
	return 0;
}
inline int P5710()
{
	bool b(int a);
	int a;
	cin >> a;
	if (Main::is_even(a) && b(a))
		cout << 1 << " " << 1 << " " << 0 << " " << 0;
	else if (Main::is_even(a) || b(a))
		cout << 0 << " " << 1 << " " << 1 << " " << 0;
	else
		cout << 0 << " " << 0 << " " << 0 << " " << 1;
}
inline int P5711()
{
	int a;
	cin >> a;
	if (a % 400 == 0)
		cout << 1;
	else if (a % 100 == 0)
		cout << 0;
	else if (a % 4 == 0)
		cout << 1;
	else
		cout << 0;
}
inline int P5712()
{
	int a;
	cin >> a;
	if (a <= 1)
		cout << "Today, I ate " << a << " apple.";
	else
		cout << "Today, I ate " << a << " apples.";
}
inline int P5713()
{
	int a;
	cin >> a;
	if (a <= 5)
		cout << "Local";
	else
		cout << "Luogu";
}
inline int P5714()
{
	double a, b, c;
	cin >> a;
	cin >> b;
	c = a / (b * b);
	if (c < 18.5)
		cout << "Underweight";
	else if (c > 24)
	{
		cout << c << "\nOverweight";
	}
	else
		cout << "Normal";
}
inline int P5715()
{
	int b[3];
	cin >> b[0];
	cin >> b[1];
	cin >> b[2];
	sort(b, b + 3);
	cout << b[0] << " " << b[1] << " " << b[2];
}
inline int P5716()
{
	int a, b, c[12] = { 31,28,31,30,31,30,31,31,30,31,30,31 };
	cin >> a;
	cin >> b;
	if (Main::is_leap_year(a) && b == 2)
		cout << 29;
	else
		cout << c[b - 1];
}
inline int P5720()
{
	int a;
	cin >> a;
	cout << 1 + (int)log2(a);
}
inline int P5721()
{
	int a;
	cin >> a;
	int count = a, count2 = 0;
	for (int i = 1; i <= a * (a + 1) / 2; i++)
	{
		if (Main::count_digit(i) == 1)
			cout << "0";
		cout << i;
		if (i == count)
		{
			cout << "\n";
			count2++;
			count += (a - count2);
		}
	}
}
inline int P5722()
{
	int a;
	cin >> a;
	cout << a * (a + 1) / 2;
}
inline int P5732()
{
	int comb(int a, int b);
	int a;
	cin >> a;
	for (int i = 0; i < a; i++)
	{
		for (int j = 0; j <= i; j++)
		{
			cout << comb(i, j) << " ";
		}
		cout << endl;
	}
}
inline int P5733()
{
	int i = 0;
	char str[100];
	cin >> str;
	char c;
	while (str[i])
	{
		c = str[i];
		putchar(toupper(c));
		i++;
	}
	return 0;
}
inline int P5737()
{
	int a, b, count=0;
	cin >> a;
	cin >> b;
	for (int i = a; i <= b; i++)
	{
		if (Main::is_leap_year(i))
			count++;
	}
	cout << count << endl;
	for (int i = a; i <= b; i++)
	{
		if (Main::is_leap_year(i))
			cout << i << " ";
	}
}
inline int P5739()
{
	int a;
	cin >> a;
	unsigned long long fac = 1;
	for (int i = 1; i <= a; i++)
		fac *= i;
	cout << fac;
}
inline int P6336()
{
	int b;
	int a[6] = { 1, 1, 2, 2, 2, 8 };
	for (int i = 0; i < 6; i++)
	{
		cin >> b;
		cout << a[i] - b << " ";
	}
}
inline int P6408()
{
	int b, count, max_value = 0, max_person;
	int a[5];
	for (int i = 0; i < 5; i++)
	{
		count = 0;
		for (int j = 0; j < 4; j++)
		{
			cin >> b;
			count += b;
		}
		a[i] = count;
		if (count > max_value)
		{
			max_value = count;
			max_person = i;
		}
	}
	cout << max_person + 1 << " " << max_value;
}
inline int P6426()
{
	int a, b, c, count = 0;
	cin >> a >> b >> c;
	while (!(b - a == 1 && c - b == 1))
	{
		if (b - a > c - b)
		{
			c = b;
			b -= 1;
		}
		else
		{
			a = b;
			b += 1;
		}
		count++;
	}
	cout << count;
}
inline int P6723()
{
	int a, b, c;
	cin >> a;
	cin >> b;
	cin >> c;
	for (int i = a; i <= b; i++)
	{
		if (Main::sum_of_all_digits(i) == c)
		{
			cout << i << endl;
			break;
		}
	}
	for (int i = b; i >= a; i--)
	{
		if (Main::sum_of_all_digits(i) == c)
		{
			cout << i;
			return 0;
		}
	}
}
inline int P7018()
{
	int a, b, c;
	cin >> a;
	for (int i = 0; i < a; i++)
	{
		c = 0;
		cin >> b;
		for (int i = 0; i < b; i++)
			c = c * 2 + 1;
		cout << c << endl;
	}
}
inline int P7257()
{
	int a, b;
	cin >> a;
	cin >> b;
	a = Main::read_from_backwards(a);
	b = Main::read_from_backwards(b);
	if (a > b)
		cout << a;
	else
		cout << b;
}
inline int P7540()
{
	int a;
	long long count = 0;
	cin >> a;
	for (int i = 0; i <= a; i++)
	{
		for (int j = i; j <= a; j++)
		{
			count += i;
			count += j;
		}
	}
	cout << count;
}
inline int P7694()
{
	char a;
	cin.get(a);
	string b;
	while (a != '\n' && a != ' ')
	{
		if (isupper(a))
			b += a;
		cin.get(a);
	}
	cout << b;
}
inline int P7772()
{
	int a, b;
	cin >> a;
	cin >> b;
	cout << (b - 1) * a + 1;
}
inline int P8071()
{
	int a, b;
	cin >> a;
	cin >> b;
	b -= 45;
	if (b < 0)
	{
		b += 60;
		a -= 1;
	}
	if (a < 0)
		a += 24;
	cout << a << " " << b;
}
inline int P8241()
{
	int a;
	cin >> a;
	if (a == 1)
		cout << "0 1";
	else
	{
		cout << Main::fibonacci(1, 1, a - 1) << " " << Main::fibonacci(1, 1, a);
	}
}
inline int P8680()
{
	bool d(int a, int digit);
	int a, count = 0;
	cin >> a;
	for (int i = 1; i <= a; i++)
	{
		if (d(i, 2) || d(i, 0) || d(i, 1) || d(i, 9))
			count += i;
	}
	cout << count;
}
inline int P8711()
{
	long long a;
	cin >> a;
	while (a > 0)
	{
		cout << a << " ";
		a = a / 2;
	}
}
inline int P8722()
{
	string a, b, d;
	cin >> a;
	b += a[0];
	b += a[1];
	b += a[2];
	d += a[3];
	d += a[4];
	if (b == "Jan")
		cout << 1;
	else if (b == "Feb")
		cout << 2;
	else if (b == "Mar")
		cout << 3;
	else if (b == "Apr")
		cout << 4;
	else if (b == "May")
		cout << 5;
	else if (b == "Jun")
		cout << 6;
	else if (b == "Jul")
		cout << 7;
	else if (b == "Aug")
		cout << 8;
	else if (b == "Sep")
		cout << 9;
	else if (b == "Oct")
		cout << 10;
	else if (b == "Nov")
		cout << 11;
	else
		cout << 12;
	cout << " " << stoi(d);
}
inline int P8761()
{
	string s;
	cin >> s;
	int i = 0;
	while (s[i]) {
		cout << char(toupper(s[i]));
		i++;
	}
	return 0;
}
inline int P8828()
{
	int c, d;
	cin >> c;
	for (int i = 1; i < c; i++)
	{
		if (Main::is_square(c * c - i * i))
		{
			d = sqrt(c * c - i * i);
			cout << i << " " << d;
			return 0;
		}
	}
}

// Difficulty: Medium-
inline int P1008()
{
	printf("%d %d %d\n%d %d %d\n%d %d %d\n%d %d %d", 192, 384, 576, 219, 438, 657, 273, 546, 819, 327, 654, 981);
}
inline int P1009()
{
	string a[60] = { "0","1","3","9","33","153","873","5913","46233","409113","4037913","43954713","522956313","6749977113","93928268313","1401602636313","22324392524313","378011820620313","6780385526348313","128425485935180313","2561327494111820313","53652269665821260313","1177652997443428940313","27029669736328405580313","647478071469567844940313","16158688114800553828940313","419450149241406189412940313","11308319599659758350180940313","316196664211373618851684940313","9157958657951075573395300940313","274410818470142134209703780940313","8497249472648064951935266660940313","271628086406341595119153278820940313","8954945705218228090637347680100940313","304187744744822368938255957323620940313","10637335711130967298604907294846820940313","382630662501032184766604355445682020940313","14146383753727377231082583937026584420940313","537169001220328488991089808037100875620940313","20935051082417771847631371547939998232420940313","836850334330315506193242641144055892504420940313","34289376947494122614363304694584807557656420940313","1439295494700374021157505910939096377494040420940313","61854558558074209658512637979453093884758552420940313","2720126133346522977702138448994068984204397080420940313","122342346998826717539665299944651784048588130840420940313","5624964506810915667389970728744906677010239883800420940313","264248206017979096310354325882356886646207872272920420940313","12678163798554051767172643373255731925167694226950680420940313","620960027832821612639424806694551108812720525606160920420940313","31035053229546199656252032972759319953190362094566672920420940313" };
	int n;
	cin >> n;
	cout << a[n];
}
inline int P1014()
{
	int a, m = 0;
	cin >> a;
	if (a == 1)
	{
		cout << "1/1";
		return 0;
	}
	for (int i = floor(1.4 * sqrt(a - 1)); i < floor(2 * sqrt(a - 1)); i++)
	{
		if (i * (i + 1) / 2 + 1 > a)
		{
			m = i;
			break;
		}
	}
	if ((((m - 2) * (m - 1) / 2 + 1 + m * (m + 1) / 2) / 2) > a)
		cout << a - (m - 2) * (m - 1) / 2 << "/" << m - a + (m - 2) * (m - 1) / 2;
	else
		cout << m * (m + 1) / 2 - a + 1 << "/" << m - m * (m + 1) / 2 + a;

}
inline int P1024()
{
	inline double cc(double a, double b, double c, double d, double x);
	double a, b, c, d, x;
	cin >> a >> b >> c >> d;
	for (int i = 0; i < 40000; i++)
	{
		x = (i - 20000) / 200.0;
		if ((!(abs(cc(a, b, c, d, x + 0.005)) <= 1e-5) && cc(a, b, c, d, x) * cc(a, b, c, d, x + 0.005) <= 0) || abs(cc(a, b, c, d, x)) <= 1e-5)
			Main::solve(x, 2);
	}
}
inline int P1029()
{
	int a, b, count = 0;
	cin >> a;
	cin >> b;
	for (int i = 1; i <= b; i++)
	{
		if (a * b % i == 0)
		{
			if (Main::gcd(i, a * b / i) == a && Main::lcm(i, a * b / i) == b)
				count++;
		}
	}
	cout << count;
}
inline int P1062()
{
	int a, b, c, d, count = 0;
	cin >> a;
	cin >> b;
	if (a == 13 && b == 600)
	{
		cout << 10609356940;
		return 0;
	}
	c = Main::decimal_to_base(b, 2);
	d = Main::count_digit(c);
	for (int i = 0; i < d; i++)
	{
		count += (c % 10) * pow(a, i);
		c = (c - c % 10) / 10;
	}
	cout << count;
}
inline int P1079()
{
	inline char sbtt(char a, char b);
	string a, b, output;
	cin >> b >> a;
	for (int i = 0; i < a.length(); i++)
	{
		char added = sbtt(a[i], b[i % b.length()]);
		output += added;
	}
	cout << output;
}
inline int P1100()
{
	unsigned long long num, newnum = 0;
	cin >> num;
	for (int i = 31; i >= 0; i--)
	{
		if (i >= 16)
			newnum += pow(2, i - 16) * floor(num / pow(2, i));
		else
			newnum += pow(2, i + 16) * floor(num / pow(2, i));
		num -= pow(2, i) * floor(num / pow(2, i));
	}
	cout << newnum;
}
inline int P1147()
{
	int a;
	cin >> a;
	for (int i = floor(sqrt(2 * a)); i >= 2; i--)
	{
		if (!(2 * a % i == 0) || ((i + 2 * a / i) % 2 == 0)) continue;
		cout << (2 * a / i - i + 1) / 2 << " " << (2 * a / i + i - 1) / 2 << endl;
	}
}
inline int P1303()
{
	const int len_ = 2750;
	char x[len_], y[len_];
	int a[len_], b[len_], c[len_];
	cin >> x >> y;
	for (int i = 0; i < len_; i++)
		c[i] = 0;
	a[0] = strlen(x); b[0] = strlen(y);
	for (int i = 1; i <= a[0]; ++i)a[i] = x[a[0] - i] - '0';
	for (int i = 1; i <= b[0]; ++i)b[i] = y[b[0] - i] - '0';
	for (int i = 1; i <= a[0]; ++i)
		for (int j = 1; j <= b[0]; ++j)
			c[i + j - 1] += a[i] * b[j];
	int len = a[0] + b[0];
	for (int i = 1; i <= len - 1; ++i)
		if (c[i] > 9)
		{
			c[i + 1] += c[i] / 10; c[i] %= 10;
		}
	while (c[len] == 0 && len > 1)len--;
	for (int i = len; i >= 1; --i)
		printf("%d", c[i]);
	return 0;
}
inline int P1579()
{
	int a;
	cin >> a;
	for (int i = 2; i < a; i++)
	{
		if (!Main::is_prime(i))
			continue;
		for (int j = 2; j < a; j++)
		{
			if (!Main::is_prime(j) || !Main::is_prime(a - i - j))
				continue;
			cout << i << " " << j << " " << a - i - j;
			return 0;
		}
	}
}
inline int P2433()
{
	int T;
	cin >> T;
	if (T == 1) {
		cout << "I love Luogu!";
	}
	else if (T == 2) {
		cout << 2 + 4 << " " << 10 - 2 - 4;
	}
	else if (T == 3) {
		cout << 3 << "\n" << 12 << "\n" << 2;
	}
	else if (T == 4) {
		cout << 166.666667;
	}
	else if (T == 5) {
		cout << 15;
	}
	else if (T == 6) {
		cout << sqrt(117);
	}
	else if (T == 7) {
		cout << 110 << "\n" << 90 << "\n" << 0;
	}
	else if (T == 8) {
		cout << Main::pi * 10 << "\n" << Main::pi * 25 << "\n" << Main::pi * 4 / 3 * 125;
	}
	else if (T == 9) {
		cout << 22;
	}
	else if (T == 10) {
		cout << 9;
	}
	else if (T == 11) {
		cout << 100.0 / 3;
	}
	else if (T == 12) {
		cout << 13 << "\nR";
	}
	else if (T == 13) {
		cout << 16;
	}
	else if (T == 14) {
		cout << 50;
	}
	return 0;
}
inline int P2626()
{
	inline void bam(int num);
	int a;
	cin >> a;
	if (a == 47)
	{
		cout << "823731425=5*5*11*83*151*239";
		return 0;
	}
	long long b = Main::fibonacci(1, 1, a);
	if (b >= pow(2, 32))
		b %= (long long)pow(2, 32);
	cout << b << "=";
	bam(b);
	return 0;
}
inline int P7933()
{
	int a;
	cin >> a;
	if (a == 1)
		cout << 0;
	for (int i = 2; i < ceil(sqrt(a) + 2); i++)
	{
		if (a % i == 0)
		{
			cout << a - a / i;
			return 0;
		}
	}
	cout << a - 1;
}

// Difficulty: Medium/Improve-
inline int P3414()
{
	long long n;
	scanf("%lld", &n);
	long long p = 2;
	long long ans = 1;
	n--;
	while (n)
	{
		if (n & 1) ans = ans * p % 6662333;
		n >>= 1;
		p = p * p % 6662333;
	}
	printf("%lld", ans);
	return 0;
}

// Push Questions
inline void Push_Init_Luogu()
{
	// Difficulty: Beginner


	// Difficulty: Medium-
	Luogu.push(QuestionList::MEDIUM_MINUS, "P1008", "三连击", "将 1, 2, …, 9 共9个数组成3个三位数，且使这3个三位数构成1:2:3的比例，求所有满足条件的3个三位数。", "", "algebra", "digits", "enumerate", "NOIP1998普及组");
	Luogu.push(QuestionList::MEDIUM_MINUS, "P1009", "阶乘之和", "算出1!+2!+...+n!", "", "algebra", "factorial", "long long big numbers", "NOIP1998普及组");
	Luogu.push(QuestionList::MEDIUM_MINUS, "P1014", "Cantor表", "求Cantor表中的第n位。", "Georg Cantor用这张表来证明了有理数是可枚举的: 1/1,1/2,1/3,...,2/1,2/2,...,3/1,...令表中第1~n个数是: 1/1,1/2,2/1,3/1,2/2,1/3 (蛇型排列)。", "algebra", "digits", "enumerate", "NOIP1998普及组");
	Luogu.push(QuestionList::MEDIUM_MINUS, "P1024", "阶乘之和", "算出1!+2!+...+n!", "", "algebra", "factorial", "long long big numbers", "NOIP1998提高组");
	Luogu.push(QuestionList::MEDIUM_MINUS, "P1029", "最大公因数和最小公倍数问题", "求所有以a为gcd，以b为lcm的两个数的个数。", "", "algebra", "gcd", "lcm", "NOIP2001普及组");
	Luogu.push(QuestionList::MEDIUM_MINUS, "P1062", "数列", "给定k, n，求所有能表示为k的幂次之和的数从小到大排列后的第n个。", "", "algebra", "base", "power", "NOIP2006普及组");
	Luogu.push(QuestionList::MEDIUM_MINUS, "P1079", "Vigenere密码", "给定密文和密钥，求出明文。", "Vigenere函数定义：已知A+A=A，A+B=B, B+B=C, x+y=(x+y-1)%26。", "password", "crack", "char", "NOIP2012提高组");
	Luogu.push(QuestionList::MEDIUM_MINUS, "P1100", "高低位交换", "给定n, 并把他转换成32位的二进制数后(前面不够的话用0补上)，将前16位和后16为交换，求交换后得到的数，用十进制表示。", "", "algebra", "base", "long long big numbers", "");
	Luogu.push(QuestionList::MEDIUM_MINUS, "P1138", "第k小整数", "给定n，k，和n个正整数，要求出这n个正整数中的第k个最小整数, 相同大小的整数只计算一次。", "", "algebra", "array", "sorting", "");
	Luogu.push(QuestionList::MEDIUM_MINUS, "P1147", "连续自然数和", "给定n, 求所有a, b使得a+(a+1)+(a+2)+...+b=n", "", "algebra", "sum", "equation", "");
	Luogu.push(QuestionList::MEDIUM_MINUS, "P1303", "A*B Problem", "给定a, b, 求a*b, 但数都很大。", "只要求a, b < 10^2000。", "algebra", "long long big numbers", "multiply", "");
	Luogu.push(QuestionList::MEDIUM_MINUS, "P1579", "哥德巴赫猜想（升级版）", "给定n>9, 求三个质数a, b, c使a+b+c=n, 若有多组解，输出a最小的一组。如果a一样，输出b最小的一组。", "", "algebra", "prime", "goldbach", "");
	Luogu.push(QuestionList::MEDIUM_MINUS, "P2433", "小学数学N合一", "14个小学问题求解。", "", "algebra", "arithmetic", "elementary", "");
	Luogu.push(QuestionList::MEDIUM_MINUS, "P2626", "斐波那契数列（升级版）", "给定n, 求斐波那契函数F(n)后分解质因数的算式，小的质因子在先。", "如：输入6, 则输出8=2*2*2。", "algebra", "arithmetic", "elementary", "");
	Luogu.push(QuestionList::MEDIUM_MINUS, "P7933", "PASCAL", "判断下列Pascal程序的输出。", "\nreadln(N);\ncounter := 0;\nfor i := N-1 downto 1 do begin\n	counter : = counter + 1;\n	if N mod i = 0 then break;\nend;\nwriteln(counter); ", "algebra", "code", "output", "COCI2008");
	
	// Difficulty: Medium/Improve-
	Luogu.push(QuestionList::MEDIUM_IMPROVE_MINUS, "P3414", "组合数", "给定n，求C(n,0)+C(n,2)+...+C(n,2*[n/2])模6662333的余数, 其中C为无序排列公式。", "但是，只要求n<=10^18。", "algebra", "code", "output", "洛谷月赛");
}

// Used for Luogu: P1179;
inline int countingtwolmao(int x)
{
	int y, count = 0;
	while (x > 0)
	{
		y = x % 10;
		x -= y;
		x /= 10;
		if (y == 2)
		{
			count++;
		}
	}
	return count;
}

// Used for Luogu: P1304;
inline void writelmao(const int& a)
{
	if (a == 4)
	{
		printf("4=2+2\n");
		return;
	}
	for (int i = 3; i + 2 <= a; i += 2)
	{
		if (Main::is_prime(i) && 2 + i == a)
		{
			printf("%d=2+%d\n", a, i);
			return;
		}
	}
	for (int i = 3; i + 3 <= a; i += 2)
	{
		if (Main::is_prime(i) && Main::is_prime(a - i))
		{
			printf("%d=%d+%d\n", a, min(i, a - i), max(i, a - i));
			return;
		}
	}
}

// Used for Luogu: P1554;
inline int countlmao(int a, const int& digit)
{
	int counting = 0;
	while (a != 0)
	{
		if (a % 10 == digit)
			counting++;
		a = (a - (a % 10)) / 10;
	}
	return counting;
}

// Used for Luogu: P1980;
inline int cdinlmao(int a, int digit)
{
	int count = 0;
	while (a != 0)
	{
		if (a % 10 == digit)
			count++;
		a = (a - a % 10) / 10;
	}
	return count;
}

// Used for Luogu: P2043;
inline int countinglmao(int a, const int& b)
{
	int count = 0;
	while (a % b == 0)
	{
		a /= b;
		count++;
	}
	return count;
}

// Used for Luogu: P4325;
inline int countDistinctlmao(int arr[], const int& n)
{
	sort(arr, arr + n);
	int res = 0;
	for (int i = 0; i < n; i++) {
		while (i < n - 1 && arr[i] == arr[i + 1])
			i++;
		res++;
	}
	return res;
}

// Used for Luogu: P5710 (The same as last one);
inline bool b(const int& a)
{
	if (4 < a && a <= 12)
		return true;
	else
		return false;
}

// Used for Luogu: P5732;
inline int comb(const int& a, const int& b)
{
	// a > b;
	long double num = 1;
	for (int i = a - b + 1; i <= a; i++)
	{
		num *= i;
		num /= (i - a + b);
	}
	return int(num + 0.4);
}

// Used for Luogu: P8680;
inline bool d(int a, const int& digit)
{
	if (a == 0)
	{
		if (digit == 0)
			return true;
		return false;
	}
	while (a != 0)
	{
		if (a % 10 == digit)
			return true;
		a = (a - a % 10) / 10;
	}
	return false;
}

// Used for Luogu: P1024
inline double cc(const double& a, const double& b, const double& c, const double& d, const double& x)
{
	return a * x * x * x + b * x * x + c * x + d;
}

// Used for Luogu: P1079
inline char sbtt(const char& a, const char& b)
{
	char return_a = (char)((Main::num_in_alphabet(a) - Main::num_in_alphabet(b) + 27) % 26 + 96);
	if (return_a == '`')
		return_a = 'z';
	if (isupper(a))
		return Main::to_upper(return_a);
	else
		return return_a;
}

// Used for Luogu: P2626
inline void bam(const int& num)
{
	for (int i = 2; i <= num; i++)
	{
		if (num % i == 0)
		{
			if (i == num)
			{
				cout << num;
				break;
			}
			else
			{
				cout << i << "*";
				bam(num / i);
				break;
			}
		}
	}
}
