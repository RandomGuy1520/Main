#pragma once
#include "Examples.h"

class Calculate
{
private:
	// Helper Functions and Variables
	double a[5]{};
	const double eps = 1e-5;
	double F(double x, int k, double y)
	{
		if (k == 1)
			return x + y;
		if (k == 2)
			return abs(x - y);
		if (k == 3)
			return x * y;
		return x / y;
	}
	char GetSign(int sign)
	{
		if (sign == 1)
			return '+';
		else if (sign == 2)
			return '-';
		else if (sign == 3)
			return '*';
		else
			return '/';
	}
	void PrintFormula(int dot1, int sign1, int dot2, int sign2, int dot3, int sign3, int dot4, int type)
	{
		if (type == 0)
			std::cout << "(((" << dot1 << GetSign(sign1) << dot2 << ")" << GetSign(sign2) << dot3 << ")" << GetSign(sign3) << dot4 << ")";
		else
			std::cout << "((" << dot1 << GetSign(sign1) << dot2 << ")" << GetSign(sign2) << "(" << dot3 << GetSign(sign3) << dot4 << "))";
	}

public:
	struct Dots
	{
		int dot1;
		int dot2;
		int dot3;
		int dot4;
		void SetDots(int a, int b, int c, int d)
		{
			dot1 = a;
			dot2 = b;
			dot3 = c;
			dot4 = d;
		}
	};
	int RandDot()
	{
		return Main::randint(1, 13);
	}
	Dots RandDots()
	{
		Dots dots;
		dots.SetDots(RandDot(), RandDot(), RandDot(), RandDot());
		return dots;
	}
	Dots RandHasSolutionDots()
	{
		Dots dots = RandDots();
		while (!HasSolutions(dots))
			dots = RandDots();
		return dots;
	}
	void PrintDots(Dots dots)
	{
		std::cout << dots.dot1 << " " << dots.dot2 << " " << dots.dot3 << " " << dots.dot4 << std::endl;
	}
	void Reset()
	{
		a[1] = 0;
		a[2] = 0;
		a[3] = 0;
		a[4] = 0;
	}
	int HasSolutions(Dots dots)
	{
		a[1] = dots.dot1;
		a[2] = dots.dot2;
		a[3] = dots.dot3;
		a[4] = dots.dot4;
		std::sort(a + 1, a + 5);
		do
		{
			for (int i = 1; i <= 4; i++)
				for (int j = 1; j <= 4; j++)
					for (int k = 1; k <= 4; k++)
						if (abs(F(F(F(a[1], i, a[2]), j, a[3]), k, a[4]) - 24) <= eps)
							return 1;
						else if (abs(F(F(a[1], i, a[2]), j, F(a[3], k, a[4])) - 24) <= eps)
							return 1;
		} while (std::next_permutation(a + 1, a + 5));
		Reset();
		return 0;
	}
	int PrintSolutions(Dots dots)
	{
		a[1] = dots.dot1;
		a[2] = dots.dot2;
		a[3] = dots.dot3;
		a[4] = dots.dot4;
		std::sort(a + 1, a + 5);
		do
		{
			for (int i = 1; i <= 4; i++)
				for (int j = 1; j <= 4; j++)
					for (int k = 1; k <= 4; k++)
						if (abs(F(F(F(a[1], i, a[2]), j, a[3]), k, a[4]) - 24) <= eps)
						{
							PrintFormula(a[1], i, a[2], j, a[3], k, a[4], 0);
							return 1;
						}
						else if (abs(F(F(a[1], i, a[2]), j, F(a[3], k, a[4])) - 24) <= eps)
						{
							PrintFormula(a[1], i, a[2], j, a[3], k, a[4], 1);
							return 1;
						}
		} while (std::next_permutation(a + 1, a + 5));
		std::cout << "No Answer!" << std::endl;
		Reset();
		return 0;
	}
};

// Condition Functions
inline bool isin(int a)
{
	if (a == 3 || a == 4 || a == 6 || a == 8)
		return true;
	else
		return false;
}

// Main Function Example
inline int Examples::dots_calculation()
{
	float a = 0;
	float total_num = 0;
	Calculate calculate;
	Calculate::Dots dots;
	for (int dot1 = 1; dot1 <= 13; dot1++)
	{
		for (int dot2 = 1; dot2 <= 13; dot2++)
		{
			for (int dot3 = 1; dot3 <= 13; dot3++)
			{
				for (int dot4 = 1; dot4 <= 13; dot4++)
				{
					dots.SetDots(dot1, dot2, dot3, dot4);
					bool has_solutions = calculate.HasSolutions(dots);
					if (true)
					{
						if (has_solutions)
							a++;
						total_num++;
						std::cout << dot1 << " " << dot2 << " " << dot3 << " " << dot4 << ": " << has_solutions << std::endl;
					}
				}
			}
		}
	}
	std::cout << std::endl << a / total_num << std::endl;
	return 0;
}

inline int Examples::dots_game()
{
	Calculate calculate;
	Calculate::Dots a;
	bool answer, has_solutions;
	while (true)
	{
		a = calculate.RandDots();
		has_solutions = calculate.HasSolutions(a);
		calculate.PrintDots(a);
		std::cin >> answer;
		if (answer != has_solutions)
			std::cout << "\033[1;31m";
		calculate.PrintSolutions(a);
		std::cout << "\033[0m";
		Main::print_endl();
	}
}
