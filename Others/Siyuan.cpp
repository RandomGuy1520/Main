#pragma once
#include "Examples.h"
#define sq(a) ((a) * (a))
#define int long long
using namespace std;

static struct node
{
	int a = 0, b = 0, c = 0;
	node() {}
	node(int a, int b, int c) : a(a), b(b), c(c) {}
};

static const enum TruthLevel { I, II, III, IV, V };
static const TruthLevel lvl = I; // Current level

static inline int GetMin()
{
	if (lvl == I) return 10;
	if (lvl == II) return 50;
	if (lvl == III) return 100;
	if (lvl == IV) return 160;
	if (lvl == V) return 220;
}

static inline int GetMax()
{
	if (lvl == I) return 50;
	if (lvl == II) return 100;
	if (lvl == III) return 160;
	if (lvl == IV) return 220;
	if (lvl == V) return 280;
}

static inline int calc(node u, node v)
{
	if (lvl == I) return abs(u.a - v.a) + abs(u.b - v.b) + abs(u.c - v.c);
	if (lvl == II) return round(sqrt(sq(u.a - v.a) + sq(u.b - v.b) + sq(u.c - v.c)));
	if (lvl == III) return (int)(u.a > v.a) + (u.b > v.b) + (u.c > v.c);
	if (lvl == IV) return (abs(u.a - v.a) ^ abs(u.b - v.b) ^ abs(u.c - v.c)) % 9 + 1;
	if (lvl == V) {
		int dx = abs(u.a - v.a), dy = abs(u.b - v.b), dz = abs(u.c - v.c);
		int md = max(max(dx, dy), dz), res = 0;
		if (dx == md) res |= 1;
		if (dy == md) res |= 2;
		if (dz == md) res |= 4;
		return res;
	}
}

static int lol[114514];

inline void Examples::Siyuan()
{
	int u, v, w, res;
	vector<node> vec;
	const int l = GetMin(), r = GetMax();
	for (int i = l; i <= r; i++)
		for (int j = l; j <= r; j++)
			for (int k = l; k <= r; k++)
				vec.push_back(node(i, j, k));
	while (cin >> u >> v >> w >> res)
	{
		vector<node> tmp;
		for (const node& q : vec)
			if (calc(node(u, v, w), q) == res)
				tmp.push_back(node(q.a, q.b, q.c));
		cout << tmp.size() << endl;
		for (int i = 0; i < min((unsigned long long)tmp.size(), 50ull); i++) cout << tmp[i].a << ' ' << tmp[i].b << ' ' << tmp[i].c << endl;
		vec = tmp;
		long long mn = 1e18; node bst;
		memset(lol, 0, sizeof(lol));
		for (int i = l; i <= r; i++)
			for (int j = l; j <= r; j++)
				for (int k = l; k <= r; k++)
				{
					memset(lol, 0, sizeof(lol));
					for (const node& q : tmp)
						lol[calc(node(i, j, k), q)]++;
					int cnt = 0;
					for (int i = 0; i <= 3 * (r - l); i++) cnt += lol[i] * lol[i]; // Heuristic
					if (cnt < mn || cnt == mn && bst.a + bst.b + bst.c > i + j + k) mn = cnt, bst = node(i, j, k);
				}
		cout << bst.a << ' ' << bst.b << ' ' << bst.c << endl;
	}
}
