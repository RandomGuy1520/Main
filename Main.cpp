#pragma once
#include "Everything.h"

namespace Main {
	const double e = 2.718281828459045;
	const double tau = 6.283185307179586;
	const double pi = 3.141592653589793;
	using namespace std::chrono;
	class cctime
	{
	private:
		time_point<steady_clock> start_time;
		time_point<steady_clock> end_time;
	public:
		typedef time_point<steady_clock> time_type;
		auto start()
		{
			start_time = high_resolution_clock::now();
			return start_time;
		}
		auto end()
		{
			end_time = high_resolution_clock::now();
			return end_time;
		}
		double duration()
		{
			std::chrono::duration<double> time_d = end_time - start_time;
			return time_d.count();
		}
	};
	class keyboard
	{
	private:
		double rest_ms = 10;
	public:
		const enum keys {
			CLICK = 1,
			LEFT_CLICK = 1,
			RIGHT_CLICK,
			BACKSPACE = 8,
			TAB,
			ENTER = 13,
			SHIFT = 16,
			CTRL,
			ALT,
			CAPS_LOCK = 20,
			ESC = 27,
			SPACEBAR = 32,
			WIN = 91,
			DEL = 46,
			LEFT_ARR = 37,
			UP_ARR,
			RIGHT_ARR,
			DOWN_ARR,
			VLM_MUTE = 173,
			VLM_DOWN,
			VLM_UP,
			K0 = 96, K1, K2, K3, K4, K5, K6, K7, K8, K9, A = 65, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
		};
		const keys letters[26] = { A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z };
		char key_to_char(keys key)
		{
			if ((int)key >= 96 && (int)key <= 105)
				return (char)((int)key - 48);
			else if ((int)key >= 65 && (int)key <= 89)
				return (char)((int)key + 32);
			else
			{
				std::cout << "Unable to convert key to char.";
				return ' ';
			}
		}
		bool is_pressed(keys x)
		{
			return GetKeyState(x) & 0x8000;
		}
		bool is_clicked()
		{
			return ((GetKeyState(CLICK) & 0x80) != 0);
		}
		void set_rest_ms(double millisec)
		{
			rest_ms = millisec;
		}
		void rest()
		{
			Sleep(rest_ms);
		}
		void rest(const double& millisec)
		{
			Sleep(millisec);
		}
		void rest(const double& probability, const double& millisec)
		{
			std::random_device rd;
			std::mt19937 mt(rd());
			std::uniform_int_distribution<int> dist(1, 10000);
			if (dist(mt) <= 10000 * probability)
				Sleep(millisec);
		}
		void keydown(const keys x)
		{
			keybd_event(x, 0, 0, 0);
		}
		void keyup(const keys x)
		{
			keybd_event(x, 0, KEYEVENTF_KEYUP, 0);
		}
		void mousedown()
		{
			mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
		}
		void mouseup()
		{
			mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
		}
		void left_mousedown()
		{
			mousedown(); // Default
		}
		void left_mouseup()
		{
			mouseup(); // Default
		}
		void right_mousedown()
		{
			mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0);
		}
		void right_mouseup()
		{
			mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0);
		}
		void hold(const keys a[], const int& length)
		{
			for (int i = 0; i < length; i++)
				keydown(a[i]);
			for (int i = length - 1; i >= 0; i--)
				keyup(a[i]);
			rest();
		}
		void upper(const keys x)
		{
			keys a[] = { SHIFT, x };
			hold(a, 2);
		}
		void upper(const int x)
		{
			upper((keys)x);
		}
		void ctrl(const keys x)
		{
			keys a[] = { CTRL, x };
			hold(a, 2);
		}
		void ctrl(const int x)
		{
			ctrl((keys)x);
		}
		void ctrl(const std::string x)
		{
			for (int i = 0; i < x.size(); i++)
			{
				ctrl(x[i]);
				Sleep(100);
			}
		}
		void write(const keys x)
		{
			keydown(x);
			keyup(x);
			rest();
		}
		void write(const int x)
		{
			write((keys)x);
		}
		void write(const char x)
		{
			if (isalpha(x))
			{
				if (isupper(x))
					upper((int)x);
				else
					write((int)(x - 32));
			}
			else if (x == '!')
				upper(49);
			else if (x == ' ')
				write(32);
			else if (x == ',')
				write(188);
			else if (x == '<')
				upper(188);
			else if (x == '.')
				write(190);
			else if (x == '>')
				upper(190);
			else if (x == '/')
				write(191);
			else if (x == '?')
				upper(191);
			else if (x == '\"')
				upper(222);
			else if (x == '\'')
				write(222);
			else if (x == ';')
				write(186);
			else if (x == ':')
				upper(186);
			else if (x == '\n')
				write(ENTER);
			else if (isdigit(x))
			{
				write((int)x);
			}
		}
		void write(const std::string x)
		{
			for (int i = 0; i < x.length(); i++)
				write(x[i]);
		}
		void click()
		{
			mousedown();
			mouseup();
			rest();
		}
		void click(int click_times)
		{
			for (int i = 0; i < click_times; i++)
				click();
		}
		void right_click()
		{
			right_mousedown();
			right_mouseup();
		}
		void right_click(int click_times)
		{
			for (int i = 0; i < click_times; i++)
				right_click();
		}
		void alphabet()
		{
			for (int i = 0; i < 26; i++)
			{
				write(letters[i]);
				rest();
			}
		}
		void upper_alphabet()
		{
			for (int i = 0; i < 26; i++)
			{
				upper(letters[i]);
				rest();
			}
		}
		void spaces_alphabet()
		{
			for (int i = 0; i < 25; i++)
			{
				write(letters[i]);
				write(SPACEBAR);
				rest();
			}
			write(Z);
		}
		void backwards_alphabet()
		{
			for (int i = 25; i >= 0; i--)
			{
				write(letters[i]);
				rest();
			}
		}
		void backwards_spaces_alphabet()
		{
			for (int i = 25; i > 0; i--)
			{
				write(letters[i]);
				write(SPACEBAR);
				rest();
			}
			write(A);
		}
	};
	class screen
	{
	public:
		struct ScreenCoords
		{
			int x;
			int y;
		};
		std::pair<int, int> get_mouse_pos()
		{
			POINT p;
			GetCursorPos(&p);
			return std::make_pair(p.x, p.y);
		}
		void print_mouse_pos_when_clicked()
		{
			keyboard keyboard;
			std::pair<int, int> last_coords = std::make_pair(-1, -1);
			while (true)
			{
				if (keyboard.is_clicked())
				{
					std::pair<int, int> a = get_mouse_pos();
					if (a.first != last_coords.first || a.second != last_coords.second)
					{
						print_mouse_pos(a);
						last_coords = a;
					}
				}
			}
		}
		void print_mouse_pos(std::pair<int, int> coords)
		{
			std::cout << coords.first << " " << coords.second << std::endl;
		}
		void print_mouse_pos()
		{
			print_mouse_pos(get_mouse_pos());
		}
		int* get_color(const int& x_coords, const int& y_coords)
		{
			HDC dng = GetDC(NULL);
			COLORREF c = GetPixel(dng, x_coords, y_coords);
			int RGB_Colors[3] = { (int)GetRValue(c), (int)GetGValue(c), (int)GetBValue(c) };
			ReleaseDC(NULL, dng);
			return RGB_Colors;
		}
		int* get_color(ScreenCoords coords)
		{
			return get_color(coords.x, coords.y);
		}
		int* get_color()
		{
			std::pair<int, int> a = get_mouse_pos();
			return get_color(a.first, a.second);
		}
		void print_color(const int* const Colors_List)
		{
			std::cout << "(";
			for (int i = 0; i < 2; i++)
				std::cout << *(Colors_List + i) << ", ";
			std::cout << *(Colors_List + 2) << ")";
		}
		void move_mouse(const int& x_coords, const int& y_coords)
		{
			SetCursorPos(x_coords, y_coords);
		}
		void move_mouse(ScreenCoords coords)
		{
			SetCursorPos(coords.x, coords.y);
		}
		void move_click(const int& x_coords, const int& y_coords)
		{
			keyboard keyboard;
			move_mouse(x_coords, y_coords);
			keyboard.rest();
			keyboard.click();
		}
		void move_click(ScreenCoords coords)
		{
			move_click(coords.x, coords.y);
		}
		void move_click(const int& x_coords, const int& y_coords, const double& wait_ms)
		{
			move_click(x_coords, y_coords);
			Sleep(wait_ms);
		}
		void move_click(const std::vector<std::pair<int, int> >& coords, const double& wait_ms)
		{
			for (int i = 0; i < coords.size(); i++)
			{
				move_click(coords[i].first, coords[i].second, wait_ms);
			}
		}
		void move_click(ScreenCoords coords, const double& wait_ms)
		{
			move_click(coords);
			Sleep(wait_ms);
		}
		void start_app(const std::string& app_name)
		{
			std::string command = "start " + app_name;
			system(command.c_str());
		}
		void start_app(const std::string& app_name, const int& rest_time_ms)
		{
			start_app(app_name);
			Sleep(rest_time_ms);
		}
		void open_website(const std::string& website_url)
		{
			keyboard keyboard;
			start_app("chrome", 1000);
			keyboard.write(website_url);
			keyboard.write(keyboard::ENTER);
			Sleep(1400);
		}
	};
	class sound
	{
	public:
		const enum notes
		{
			C, C_Sharp, D_Flat = 1, D, D_Sharp, E_Flat = 3,
			E, F, F_Sharp, G_Flat = 6, G, G_Sharp, A_Flat = 8, A, A_Sharp, B_Flat = 10, B
		};
		struct Notes
		{
			notes note;
			int num;
			Notes() = default;
			Notes(notes note, int num) : note(note), num(num) {}
			int GetVal() const
			{
				return num * 12 + note;
			}
			bool operator<(Notes other)
			{
				return num < other.num ? true : num == other.num && note < other.note;
			}
			bool operator>(Notes other)
			{
				return num > other.num ? true : num == other.num && note > other.note;
			}
			bool operator==(Notes other)
			{
				return note == other.note && num == other.num;
			}
			Notes operator+(int val)
			{
				Notes ret{};
				int _val = GetVal() + val;
				ret.num = _val / 12; ret.note = notes(_val % 12);
				return ret;
			}
		};
		struct Music
		{
			notes note;
			int num;
			int sec;
			Music() = default;
			Music(Notes nt, int sec) : note(nt.note), num(nt.num), sec(sec) {}
			Music(notes note, int num, int sec) : note(note), num(num), sec(sec) {}
		};
		const double half = 1.0594630943592953;
		const double two_half = pow(half, 2);
		const int octave = 2;
		const Notes LEFT_NOTE = { A, 1 };
		const Notes MIDDLE_C = { C, 4 };
		const Notes RIGHT_NOTE = { C, 8 };
		double freq(notes note, int num)
		{
			if ((num == 0 && (note == A || note == B || note == A_Sharp)) || (num == 8 && note == C) || (1 <= num && num <= 7))
				return 27.5 * pow(half, note + 3) * pow(octave, num - 1);
			perror("ERROR: Main::sound::Notes.note -> Note not found");
			return 0;
		}
		double freq(notes note)
		{
			return freq(note, 4);
		}
		double freq(Notes note)
		{
			return freq(note.note, note.num);
		}
		void set_volume(int volume)
		{
			keyboard keyboard;
			keyboard.write(keyboard.VLM_MUTE);
			for (int i = 0; i < (int)(volume / 2); i++)
				keyboard.write(keyboard.VLM_UP);
		}
		void play(int frequency, int seconds)
		{
			Beep(frequency, 1000 * seconds);
		}
		void play(Music music)
		{
			play(freq(music.note, music.num), music.sec);
		}
	};
	class keybd_piano : sound
	{
	private:
		std::list<std::tuple<Notes, std::thread*, bool> > playing_notes; // in order
		std::unordered_map<char, Notes> mp;
	public:
		keybd_piano()
		{
			std::vector<char> keybd = { 'Q', 'W', 'E', 'R', 'T', 'Y', 'U',
										'A', 'S', 'D', 'F', 'G', 'H', 'J',
										'Z', 'X', 'C', 'V', 'B', 'N', 'M' };
			std::vector<notes> octa = { C, D, E, F, G, A, B };
			/*for (int i = 0; i < 3; i++)
				for (int j = 0; j < 7; j++)
					mp[cnt + keybd[7 * i + j]] = Notes(octa[j], i + 3);*/
		}
		std::vector<Notes> KeyToNotes(std::vector<int> keys)
		{
			std::vector<Notes> res;
			/*for (int i : keys) res.push_back(mp[cnt + i]);*/
			return res;
		}
		//void PlayKeys(std::vector<Notes> keys)
		//{
		//	std::vector<Notes> start, end;
		//	for (std::tuple<Notes, std::thread*, bool>& tp : playing_notes)
		//	{
		//		Notes j = std::get<0>(tp);
		//		bool t = false;
		//		for (Notes i : keys)
		//			if (i == j)
		//				t = true;
		//		if (!t)
		//		{
		//			std::get<2>(tp) = false;
		//			std::get<1>(tp)->join();
		//			playing_notes.erase(&tp);
		//		}
		//	}
		//}
	};
	inline static int remainder(const int& a, const int& b)
	{
		return a % b;
	}
	inline static int plus(const int& a, const int& b)
	{
		return a + b;
	}
	inline static int minus(const int& a, const int& b)
	{
		return a - b;
	}
	inline static int abs_minus(const int& a, const int& b)
	{
		return abs(a - b);
	}
	inline static int times(const int& a, const int& b)
	{
		return a * b;
	}
	inline static double float_section(const float& a)
	{
		return a - floor(a);
	}
	inline static double divide(const float& a, const float& b)
	{
		return a / b;
	}
	inline static double lg(const long long& a)
	{
		return log10(a);
	}
	inline static double ln(const long long& a)
	{
		return log(a);
	}
	inline static double square_root(const int& a)
	{
		return sqrt(a);
	}
	inline static double loga_to_base(const long long& a, const int& base)
	{
		return log(a) / log(base);
	}
	inline static int max_log(const int& a, const int& base)
	{
		return floor(loga_to_base(a, base));
	}
	inline static int max_pow(const int& a, const int& base)
	{
		return pow(a, max_log(a, base));
	}
	inline static int down_divide(const float& a, const float& b)
	{
		return floor(a / b);
	}
	inline static int up_divide(const float& a, const float& b)
	{
		return ceil(a / b);
	}
	inline static int fo_up_fi_down(const float& a)
	{
		if (float_section(a) < 0.5)
			return floor(a);
		else
			return ceil(a);
	}
	inline static int fo_up_fi_down_divide(const float& a, const float& b)
	{
		return fo_up_fi_down(a / b);
	}
	inline static int factor_count(const int& a)
	{
		// including 1 and a
		int count = 0;
		for (int i = 1; i * i <= a; i++)
		{
			if (a % i == 0)
				count++;
		}
		return 2 * count - (pow(floor(sqrt(a)), 2) == a);
	}
	inline static int factor_sum(const int& a)
	{
		// including 1 and a
		int sum = 0, i;
		for (i = 1; i * i < a; i++)
		{
			if (a % i == 0)
				sum += (i + a / i);
		}
		if (i * i == a) sum += i;
		return sum;
	}
	inline static int collatz(const int& a)
	{
		if (a % 2 == 0)
			return (a / 2);
		else
			return (3 * a + 1);
	}
	inline static int sum(int* begin, int* end)
	{
		int sum = 0;
		for (int* i = begin; i < end; i++)
			sum += *i;
		return sum;
	}
	inline static double mean(int* begin, int* end)
	{
		return sum(begin, end) / (end - begin);
	}
	inline static double median(int* begin, int* end)
	{
		int* _begin, * _end, ans;
		for (int* i = _begin; i < _end; i++)
			*i = *(begin + (i - _begin));
		std::sort(begin, end);
		int len = end - begin;
		if (len % 2 == 1)
			ans = *(begin + (len - 1) / 2);
		else
			ans = (double)(*(begin + len / 2) + *(begin + len / 2 - 1)) / 2.0;
		for (int* i = begin; i < end; i++)
			*i = *(_begin + (i - begin));
		return ans;
	}
	inline static int mode(const int& len, const int a[])
	{
		int maxvalue = 0, maxcount = 0;
		for (int i = 0; i < len; i++)
		{
			int count = 0;
			for (int j = 0; j < len; j++)
			{
				if (*(a + j) == *(a + i))
					++count;
			}
			if (count > maxcount)
			{
				maxcount = count;
				maxvalue = *(a + i);
			}
			else if (maxcount == 1)
				maxvalue = -1;
		}
		return maxvalue;
	}
	inline static int gcd(const int& a, const int& b)
	{
		for (int i = min(a, b); i > 0; i--)
		{
			if (a % i == 0 and b % i == 0)
				return i;
		}
		return 1;
	}
	inline static int lcm(const int& a, const int& b)
	{
		return a * b / gcd(a, b);
	}
	inline static int arithmetic_sequence(const int& start1, const int& diff, const int& numberx)
	{
		// numberx includes start1
		return (numberx - 1) * diff + start1;
	}
	inline static int geometric_sequence(const int& start1, const int& ratio, const int& numberx)
	{
		//numberx includes start1
		return pow(ratio, numberx - 1) * start1;
	}
	inline static int fibonacci(int start1, int start2, int numberx)
	{
		int a;
		if (numberx == 1)
			return start1;
		else if (numberx == 2)
			return start2;
		for (int i = 0; i < numberx - 2; i++)
		{
			a = start1 + start2;
			start1 = start2;
			start2 = a;
		}
		return start2;
	}
	inline static long long factorial(const int& a)
	{
		long long fac = 1;
		for (int i = 1; i <= a; i++)
			fac *= i;
		return fac;
	}
	inline static long long combination(const int& a, const int& b)
	{
		long double sum = 1;
		int c = b;
		if (c > a / 2)
			c = a - b;
		for (int i = 1; i <= c; i++)
			sum = sum * (a - c + i) / i;
		return round(sum);
	}
	inline static long long catalan(const int& a)
	{
		long double count = 1;
		for (double i = 2.0; i <= a; i++)
			count *= (a + i) / i;
		if (a == 30)
			count += 1;
		return (long long)(round(count));
	}
	inline static long long has_digit_num(const int& n, const int& digit)
	{
		// Calculates how many numbers from 1 to n has the digit 'digit'.
		if (n == 0) return 0;
		int k = n / max_pow(n, 10);
		return (k > digit ? k - 1 : k) * (long long)pow(9, floor(log10(n))) + (k == digit ? -1 : has_digit_num(n % max_pow(n, 10), digit));
	}
	inline static int count_digit(long long n)
	{
		if (n == 0)
			return 1;
		int count = 0;
		while (n != 0) {
			n /= 10;
			count++;
		}
		return count;
	}
	inline static int remove_last_digit(const int& a)
	{
		return a / 10;
	}
	inline static int add_digit_at_back(const int& a, const int& digit)
	{
		return 10 * a + digit;
	}
	inline static int read_from_backwards(int a)
	{
		int new_num = 0;
		while (a != 0)
		{
			new_num = 10 * new_num + a % 10;
			a /= 10;
		}
		return new_num;
	}
	inline static int sum_of_all_digits(int a)
	{
		int sum = 0;
		while (a != 0)
		{
			sum += a % 10;
			a /= 10;
		}
		return sum;
	}
	inline static int decimal_to_base(int num, const int& base)
	{
		if (base == 10)
			return num;
		else if (base > 10)
		{
			std::cout << "Error: Can't convert to base greater than 10";
			return 0;
		}
		int converted = 0, maxpow = floor(loga_to_base(num, base));
		for (int i = maxpow; i >= 0; i--)
		{
			converted = converted * 10 + floor(num / pow(base, i));
			num -= floor(num / pow(base, i)) * pow(base, i);
		}
		return converted;
	}
	inline static int base_to_decimal(const std::string& num, const int& base)
	{
		int count = 0;
		for (int i = 0; i < num.length(); i++)
			count += pow(base, num.length() - i - 1) * ((int)num[i] - (isalpha(num[i]) ? 55 : 48));
		return count;
	}
	inline static int base_to_decimal(const int& num, const int& base)
	{
		return base_to_decimal(std::to_string(num), base);
	}
	inline static int randint(const int& startnum, const int& endnum)
	{
		// range includes startnum and endnum
		std::random_device rd;
		std::mt19937 mt(rd());
		std::uniform_int_distribution<int> dist(startnum, endnum);
		return dist(mt);
	}
	inline static int char_to_digit(const char& a)
	{
		return (int)a - 48;
	}
	inline static int num_in_alphabet(const char& a)
	{
		return (int)(isupper(a) ? (char)((int)a + 32) : a) - 96;
	}
	inline static int occurrence_of_digit(long long a, const int& digit)
	{
		int count = 0;
		while (a != 0)
		{
			if (a % 10 == digit)
				count++;
			a /= 10;
		}
		return count;
	}
	inline static int occurrence(const std::string& a, const std::string& b)
	{
		int occurrences = 0;
		std::string::size_type pos = 0;
		while ((pos = a.find(b, pos)) != std::string::npos) {
			++occurrences;
			pos += 1;
		}
		return occurrences;
	}
	inline static bool probability(const double& probability)
	{
		const int times = 10000;
		if (randint(1, times) <= probability * times)
			return true;
		else
			return false;
	}
	inline static bool has_digit(int a, const int& digit)
	{
		if (a == 0)
			return (digit == 0 ? true : false);
		while (a != 0)
		{
			if (a % 10 == digit)
				return true;
			a /= 10;
		}
		return false;
	}
	inline static bool is_int(const float& a)
	{
		if (a == int(a))
			return true;
		else
			return false;
	}
	inline static bool is_digit(const int& a)
	{
		if (a >= 0 && a <= 9)
			return true;
		else
			return false;
	}
	inline static bool is_digit(const char& a)
	{
		return isdigit(a);
	}
	inline static bool is_prime(const int& a)
	{
		if (a == 0 || a == 1)
			return false;
		for (int i = 2; i * i <= a; i++)
			if (a % i == 0)
				return false;
		return true;
	}
	inline static bool is_composite(const int& a)
	{
		if (a == 0 || a == 1)
			return false;
		return !is_prime(a);
	}
	inline static bool is_perfect(const int& a)
	{
		if (a == 0 or a == 1 or (factor_sum(a) != 2 * a))
			return false;
		else
			return true;
	}
	inline static bool is_leap_year(const int& a)
	{
		if (a % 400 == 0)
			return true;
		else if (a % 100 == 0)
			return false;
		else if (a % 4 == 0)
			return true;
		else
			return false;
	}
	inline static bool is_palindrome(const int& a)
	{
		if (a == read_from_backwards(a))
			return true;
		else
			return false;
	}
	inline static bool is_palindrome(const std::string& a)
	{
		for (int i = 0; i < a.size() / 2; i++)
		{
			if (a[i] != a[a.size() - 1 - i])
				return false;
		}
		return true;
	}
	inline static bool is_square(const int& a)
	{
		int h = a & 0xf;
		if (h != 0 && h != 1 && h != 4 && h != 9)
			return false;
		return pow(floor(sqrt(a)), 2) == a;
	}
	inline static bool is_odd(const int& a)
	{
		return (a % 2 == 0) ? false : true;
	}
	inline static bool is_even(const int& a)
	{
		return (a % 2 == 0) ? true : false;
	}
	inline static bool contains(const std::string& a, const std::string& b)
	{
		return a.find(b) != std::string::npos;
	}
	inline static bool contains_either_way(const std::string& a, const std::string& b)
	{
		return contains(a, b) || contains(b, a);
	}
	inline static bool is_in_array(const int& len, const int& a, int b[])
	{
		std::sort(b, b + len);
		return std::binary_search(b, b + len, a);
	}
	inline static bool is_in_range(const int& num, const int& lower, const int& upper)
	{
		return (lower <= num && num <= upper) ? true : false;
	}
	inline static double one_one_equation(const float& a, const float& b)
	{
		// ax = b
		// x = b / a
		return b / a;
	}
	inline static std::pair<double, double> two_one_equation(const float& a, const float& b, const float& c, const float& d, const float& e, const float& f)
	{
		// ax + by = c
		// dx + ey = f
		// x = (ce-bf)/(ae-bd)
		// y = (cd-af)/(bd-ae)
		return std::make_pair((c * e - b * f) / (a * e - b * d), (c * d - a * f) / (b * d - a * e));
	}
	inline static char digit_to_char(const int& digit)
	{
		return (char)(digit+48);
	}
	inline static char to_lower(const char& a)
	{
		if (isupper(a))
			return (char)((int)a + 32);
		else
			return a;
	}
	inline static char to_upper(const char& a)
	{
		if (islower(a))
			return (char)((int)a - 32);
		else
			return a;
	}
	inline static char random_char(bool exclude_symbols = false, bool exclude_numbers = false)
	{
		int rand_int;
		if (!exclude_symbols && !exclude_numbers)
			rand_int = randint(43, 126);
		else if (exclude_symbols && exclude_numbers)
		{
			rand_int = randint(71, 122);
			if (rand_int < 'a') rand_int -= 6;
		}
		else if (exclude_symbols && !exclude_numbers)
		{
			rand_int = randint(61, 122);
			if (rand_int < 71) rand_int -= 13;
			else if (rand_int < 97) rand_int -= 6;
		}
		else
		{
			rand_int = randint(43, 126);
			if (rand_int <= '9') rand_int -= 10;
		}
		return (char)rand_int;
	}
	inline static std::string str_first_nth(const std::string& a, const int& b)
	{
		return a.substr(0, b);
	}
	inline static std::string tostr(int a)
	{
		std::string res;
		while (a != 0) res += (a % 10 + '0'), a /= 10;
		std::reverse(res.begin(), res.end());
		return res;
	}
	inline static std::string to_lower(const std::string& a)
	{
		std::string return_str;
		for (int i = 0; i < a.length(); i++)
			return_str += to_lower(a[i]);
		return return_str;
	}
	inline static std::string to_upper(const std::string& a)
	{
		std::string return_str;
		for (int i = 0; i < a.length(); i++)
			return_str += to_upper(a[i]);
		return return_str;
	}
	inline static std::string random_string(const int& length, bool exclude_symbols = false, bool exclude_numbers = false)
	{
		std::string str;
		for (int i = 0; i < length; i++)
			str += random_char(exclude_symbols, exclude_numbers);
		return str;
	}
	inline static std::wstring to_wstring(std::string str)
	{
		std::wstring wideString(str.begin(), str.end());
		return wideString;
	}
	inline static std::vector<int> tovec(int* begin, int* end)
	{
		std::vector<int> ret;
		for (int* i = begin; i < end; i++)
			ret.push_back(*i);
		return ret;
	}
	inline static std::vector<int> get_digits(const int& a)
	{
		std::vector<int> res; int k = a;
		while (k != 0) res.push_back(k % 10), k /= 10;
		std::reverse(res.begin(), res.end());
		return res;
	}
	inline static std::vector<int> get_factors(const int& a)
	{
		// including 1 and a
		std::vector<int> res;
		for (int i = 1; i * i <= a; i++)
		{
			if (a % i == 0)
			{
				res.push_back(i);
				if (i * i != a) res.push_back(a / i);
			}
		}
		return res;
	}
	inline static std::vector<int> get_prime_factors(const int& a)
	{
		// including 1 and a
		std::vector<int> res;
		for (int i = 1; i * i <= a; i++)
		{
			if (a % i == 0 && is_prime(i))
			{
				res.push_back(i);
				if (i * i != a && is_prime(a / i)) res.push_back(a / i);
			}
		}
		return res;
	}
	inline static void taking_notes()
	{
		std::string a;
		std::cin >> a;
		while (a != "break")
			std::cin >> a;
	}
	inline static void print_endl()
	{
		std::cout << std::endl;
	}
	inline static void sleep(const int& sec)
	{
		Sleep(1000 * sec);
	}
	inline static void solve(const double& number, const int& place) {
		std::cout << std::fixed;
		std::cout << std::setprecision(place);
		std::cout << number << std::endl;
	}
	inline static void guess_the_number(const int& startnum, const int& endnum)
	{
		int turns = abs(floor((log2(endnum - startnum) - 1) * 0.8)), guessnum = randint(startnum, endnum), input;
		std::string input2;
		std::cout << "Let's play a game called guess_the_number!\n";
		std::cout << "In this game, I am thinking of a number, and you have " << turns << " turns to guess it.\n";
		std::cout << "In each turn, you can guess a number, and I will say whether it is too big or too small.\n";
		std::cout << "See if you can guess it! ";
		for (int turn = 1; turn <= turns; turn++)
		{
			std::cout << "Now, please guess a number.\n";
			if (turn == turns)
			{
				std::cout << "This is your last guess.Guess wisely.\n";
				std::cin >> input;
				if (input == guessnum)
					std::cout << "Wow! Congratulations, you guessed it on your last try! You WIN!!!\n";
				else
				{
					std::cout << "Nope.This number is not correct, and you have ran out of guesses.Game Over.You lose.\n";
					std::cout << "The number is " << guessnum << ".\n";
				}
			}
			else
			{
				std::cin >> input;
				if (input < guessnum)
					std::cout << "Too small! You still have " << turns - turn << " tries.";
				else if (input > guessnum)
					std::cout << "Too big! You still have " << turns - turn << " tries.";
				else
					std::cout << "Congratulations, that is the number I'm thinking! You WIN!!!\n";
			}
		}
		std::cout << "Do you want to play again? \n";
		std::cin >> input2;
		if (input2 == "Yes" or input2 == "yes" or input2 == "YES")
			guess_the_number(startnum, endnum);
	}
}
