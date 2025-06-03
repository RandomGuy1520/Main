#pragma once
#include "Examples.h"
using namespace std;
using namespace Main;

const int SCREEN_X = 1920, SCREEN_Y = 1080;
inline bool is_init = false;

inline string next(string k)
{
	if (k[k.size() - 1] == 'z')
		return next(k.substr(0, k.size() - 1)) + 'a';
	else return k.substr(0, k.size() - 1) + (char)(k[k.size() - 1] + 1);
}

inline void init()
{
	if (!is_init) return;
	Sleep(5000);
	keyboard keyboard;
	keyboard.ctrl('T');
	Sleep(500);
	keyboard.write("note.ms/nevergonnagiveyouup\n");
	Sleep(1500);
	keyboard.ctrl("ACW");
}

inline void hyperlinks(int num)
{
	Sleep(5000);
	string last_string;
	keyboard keyboard;
	keyboard.ctrl("AX");
	keyboard.write("Your Note.ms will be in this stream of links. Through " + to_string(num) + " links will be your Note.\n");
	for (int i = 0; i < num; i++)
	{
		last_string = random_string(8, true, true);
		cout << "note.ms/" << last_string << endl;
		keyboard.write("note.ms/" + last_string);
		Sleep(300);
		keyboard.ctrl('T');
		Sleep(300);
		keyboard.write("note.ms/" + last_string + '\n');
		Sleep(2000);
	}
	keyboard.ctrl('V');
}

inline void hackccp()
{
	init();
	vector<string> a = { "china", "China", "CHINA", "Chinese", "chinese", "maozedong", "zhouenlai", "MZD", "ZEL", "zhongguogongchandang", "CN", "cn", "aiguo", "dabaoxiaorizi", "gongchanzhuyiwansui", "ROC", "RPC", "Chinesesocialism", "CommunistParty", "CarlMarx", "dadaoribendiguozhuyi" };
	keyboard keyboard;
	for (int i = 0; i < a.size(); i++)
	{
		keyboard.ctrl('T');
		Sleep(500);
		keyboard.write("note.ms/" + a[i] + '\n');
		Sleep(2000);
		keyboard.ctrl('A');
		keyboard.write(keyboard::BACKSPACE);
		keyboard.ctrl('V');
		Sleep(200);
	}
}

inline void loopnum(int start = 116, int end = 1000)
{
	init();
	keyboard keyboard;
	for (int k = start; k < end; k++)
	{
		keyboard.ctrl('T');
		Sleep(500);
		keyboard.write("note.ms/" + to_string(k) + '\n');
		Sleep(2500);
		keyboard.ctrl('W');
	}
}

inline void loopstr(string start = "aaaa", string end = "zzzz")
{
	init();
	keyboard keyboard;
	for (string k = start; k != end; k = next(k))
	{
		keyboard.ctrl('T');
		Sleep(500);
		keyboard.write("note.ms/" + k + '\n');
		Sleep(2500);
		keyboard.ctrl('W');
	}
}

inline void loopblank()
{
	init();
	keyboard keyboard;
	while (true)
	{
		keyboard.ctrl('T');
		Sleep(300);
		keyboard.write("note.ms\n");
		Sleep(2500);
		keyboard.ctrl('V');
		Sleep(300);
	}
}
