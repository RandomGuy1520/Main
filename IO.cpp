#pragma once
#include "Everything.h"

namespace IO
{
	template <typename T>
	inline void read(T& value)
	{
		if constexpr (std::is_integral_v<T>)
		{
			int x = 0, f = 1;
			char ch = getchar();
			while (!isdigit(ch)) {
				if (ch == '-')
					f = -1;
				ch = getchar();
			}
			while (isdigit(ch)) {
				x = (x << 1) + (x << 3) + (ch ^ 48);
				ch = getchar();
			}
			value = x * f;
		}
		else std::cin >> value;
	}
	template <typename T, typename... Targs>
	static inline void read(T& value, Targs&... Fargs)
	{
		read(value);
		read(Fargs...);
	}
	template <typename T>
	static inline void read(T* begin, T* end)
	{
		for (T* i = begin; i < end; i++)
			read(*i);
	}
	template <class Iter>
	static inline void read(Iter begin, Iter end)
	{
		for (Iter i = begin; i < end; i++)
			read(*i);
	}
	template <typename T>
	static inline void print(T value)
	{
		if constexpr (std::is_integral_v<T>)
		{
			if (value < 0) putchar('-'), value = -value;
			if (value > 9) print(value / 10);
			putchar(value % 10 + '0');
		}
		else std::cout << value;
	}
	template <typename T>
	static void println(T value)
	{
		print(value); puts("");
	}
	template <typename T>
	static void print_space(T value)
	{
		print(value); putchar(' ');
	}
	template <typename T, typename... Targs>
	static void print(T value, Targs... Fargs)
	{
		print(value);
		print(Fargs...);
	}
	template <typename T>
	static void print(T* begin, T* end)
	{
		for (T* i = begin; i < end; i++)
			print(*i);
	}
	template <class Iter>
	static void print(Iter begin, Iter end)
	{
		for (Iter i = begin; i < end; i++)
			print(*i);
	}
	template <typename T, typename... Targs>
	static void print_space(T value, Targs... Fargs)
	{
		print_space(value);
		print_space(Fargs...);
	}
	template <typename T>
	static void print_space(T* begin, T* end)
	{
		for (T* i = begin; i < end; i++)
			print_space(*i);
	}
	template <class Iter>
	static void print_space(Iter begin, Iter end)
	{
		for (Iter i = begin; i < end; i++)
			print_space(*i);
	}
	template <typename T, typename... Targs>
	static void println(T value, Targs... Fargs)
	{
		println(value);
		println(Fargs...);
	}
	template <typename T>
	static void println(T* begin, T* end)
	{
		for (T* i = begin; i < end; i++)
			println(*i);
	}
	template <class Iter>
	static void println(Iter begin, Iter end)
	{
		for (Iter i = begin; i < end; i++)
			println(*i);
	}
}
