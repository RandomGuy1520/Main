#pragma once
#define MACROS
#ifdef  MACROS
#define ll long long
#define ull unsigned long long
#define desync() ios::sync_with_stdio(false), cin.tie(nullptr), cout.tie(nullptr);
#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))
#define repeat(j)      for (int i = 0; i <  j; i++)
#define iter(i, a, b)  for (int i = a; i <= b; i++)
#define iterzero(i, j) for (int i = 0; i <  j; i++)
#define iterone(i, j)  for (int i = 1; i <= j; i++)
#define itervec(i, j)  for (auto& i : j)
#define square(i)      ((i) * (i))
#define input(i)       std::cin >> (i)
#define outprecision(i) std::cout << std::fixed << std::setprecision(i);
#endif
