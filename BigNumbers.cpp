#include "Everything.h"
#include "OI.cpp"

namespace BigInt
{
	class Int
	{
	private:
		std::vector<int> nums;
		int len;
		static const int MAXN = 2e6 + 7;
		void check_len()
		{
			if (nums.size() <= len)
				nums.resize(len + (len >> 1));
		}
	public:
		operator int()
		{
			int res = 0;
			for (int i = len - 1; i >= 0; i--)
				res = 10 * res + nums[i];
			return res;
		}
		Int() { len = 0; nums.resize(MAXN); }
		Int(std::string s)
		{
			len = s.size();
			nums.resize(max(MAXN, len + 1), 0);
			reverse(s.begin(), s.end());
			for (int i = 0; i < len; i++)
				nums[i] = s[i] - '0';
		}
		Int(int n)
		{
			*this = n;
		}
		Int(const std::vector<int>& vec)
		{
			nums = vec;
			len = nums.size();
			if (nums.size() < MAXN) nums.resize(MAXN);
		}
		void SetLength(int length) { nums.resize(length); }
		void print()
		{
			for (int i = len - 1; i >= 0; i--)
				std::cout << nums[i];
		}
		bool operator>(const Int& other)
		{
			if (len > other.len)
				return true;
			for (int i = max(len, other.len) - 1; i >= 0; i--)
				if (nums[i] != other.nums[i])
					return nums[i] > other.nums[i];
			return false;
		}
		bool operator<(const Int& other)
		{
			if (len < other.len)
				return true;
			for (int i = max(len, other.len) - 1; i >= 0; i--)
				if (nums[i] != other.nums[i])
					return nums[i] < other.nums[i];
			return false;
		}
		bool operator==(const Int& other)
		{
			return nums == other.nums && len == other.len;
		}
		bool operator<=(const Int& other)
		{
			return *this < other || *this == other;
		}
		bool operator>=(const Int& other)
		{
			return *this > other || *this == other;
		}
		bool operator!=(const Int& other)
		{
			return !(*this == other);
		}
		Int operator=(const Int& other)
		{
			nums = other.nums;
			len = other.len;
			return *this;
		}
		Int operator+(const Int& other) const
		{
			Int res;
			res.len = max(len, other.len);
			res.check_len();
			int carry = 0;
			for (int i = 0; i < res.len; i++)
			{
				res.nums[i] = nums[i] + other.nums[i] + carry;
				carry = res.nums[i] / 10;
				res.nums[i] %= 10;
			}
			if (carry > 0)
			{
				res.len++;
				res.check_len();
				res.nums[res.len - 1] = carry;
			}
			return res;
		}
		Int operator+=(const Int& other)
		{
			return (*this) = (*this) + other;
		}
		Int operator-(const Int& b) const
		{
			Int res;
			res.len = max(len, b.len);
			res.check_len();
			for (int i = 0; i < res.len; i++)
			{
				if (res.nums[i] < b.nums[i])
				{
					res.nums[i + 1]--;
					res.nums[i] += 10;
				}
				res.nums[i] = nums[i] - b.nums[i];
			}
			while (res.nums[res.len - 1] == 0 && res.len > 1)
				res.len--;
			return res;
		}
		Int operator-=(const Int& b)
		{
			return (*this) = (*this) - b;
		}
		Int operator*(const Int& b) const
		{
			OI::Algorithms alg;
			std::vector<int> _a, _b;
			for (int i = 0; i < len; i++) _a.push_back(nums[i]);
			for (int i = 0; i < b.len; i++) _b.push_back(b.nums[i]);
			std::reverse(_a.begin(), _a.end());
			std::reverse(_b.begin(), _b.end());
			std::vector<int> out = alg.FFT(_a, _b);
			for (int i = out.size() - 1; i >= 1; i--)
			{
				out[i - 1] += out[i] / 10;
				out[i] = out[i] % 10;
			}
			std::reverse(out.begin(), out.end());
			if (out.back() > 9) out.push_back(out.back() / 10), out[out.size() - 2] = out[out.size() - 2] % 10;
			return Int(out);
		}
		Int operator*=(const Int& b)
		{
			return (*this) = (*this) * b;
		}
		Int operator/(int b) const
		{
			Int res;
			res.len = len;
			res.check_len();
			int r = 0;
			for (int i = len; i >= 0; i--) {
				r = r * 10 + nums[i];
				res.nums[i] = r / b;
				r %= b;
			}
			while (res.nums[res.len - 1] == 0 && res.len > 1) res.len--;
			return res;
		}
		Int operator%(int b) const
		{
			return (*this) - (*this) / b * (Int)b;
		}
		Int operator++(signed)
		{
			Int temp("1");
			*this += temp;
			return *this; // for prefix increment
		}
		Int& operator++()
		{
			*this += Int("1");
			return *this;
		}
		Int operator--(signed)
		{
			Int temp("1");
			*this -= temp;
			return *this;
		}
		Int& operator--()
		{
			*this -= Int("1");
			return *this;
		}
	};
	static std::ostream& operator<<(std::ostream& out, Int a)
	{
		a.print();
		return out;
	}
	static std::istream& operator>>(std::istream& in, Int& a)
	{
		std::string t;
		std::cin >> t;
		Int b(t);
		a = b;
		return in;
	}
}
