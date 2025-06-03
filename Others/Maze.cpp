#pragma once
#include "Examples.h"

static const std::string LogFileLoc = "MazeGame.log";
static const int MAXLEVELS = 6;
static int WAITING_TIME_MS = 20;
static int m = 15, n = 15, plx = 1, ply = 0, x_num = 1, y_num = 1;
static bool SPEEDY = 0, FULLSCREEN = 1, ADV[MAXLEVELS];

static std::vector<int> block_row, block_col, block_dir;
static std::vector<std::pair<int, int>> MazeSize = { {101, 59}, {15, 15}, {25, 25}, {45, 45}, {65, 59}, {85, 59} };
static std::mt19937 rd(std::chrono::system_clock::now().time_since_epoch().count());
static enum status { WALL, ROAD };
static enum direction { DOWN = 1, RIGHT = 2, LEFT = 4, UP = 8 };
static status a[105][105];

static class Maze
{
public:
	inline void Init() const { memset(a, WALL, sizeof(a)); }
	inline void Print() const
	{
		for (int i = 0; i <= m + 1; i++)
		{
			for (int j = 0; j <= n + 1; j++)
				if (i == plx && j == ply) printf("△");
				else if (i == m && j == n + 1) printf("☆");
				else if (i == 1 && j == 0 || a[i][j] == ROAD) printf("  ");
				else printf("█ ");
			printf("\n");
		}
	}
	inline int Get(int x, int y) const { return a[x][y]; }
	static inline void Change(int x, int y)
	{
		a[x][y] = ROAD;
	}
} maze;
static class IO {} out;

static inline void FullScreen()
{
	HWND hwnd = GetForegroundWindow();
	int cx = GetSystemMetrics(SM_CXSCREEN);
	int cy = GetSystemMetrics(SM_CYSCREEN);
	LONG l_WinStyle = GetWindowLong(hwnd, GWL_STYLE);
	SetWindowLong(hwnd, GWL_STYLE, (l_WinStyle | WS_POPUP | WS_MAXIMIZE) & ~WS_CAPTION & ~WS_THICKFRAME & ~WS_BORDER);
	SetWindowPos(hwnd, HWND_TOP, 0, 0, cx, cy, 0);
}
static inline void Cancel()
{
	HWND hWnd = GetConsoleWindow();
	LONG style = GetWindowLong(hWnd, GWL_STYLE);
	style |= WS_CAPTION;
	style |= WS_THICKFRAME;
	style |= WS_BORDER;
	style &= ~WS_POPUP;
	style &= ~WS_MAXIMIZE;
	SetWindowLong(hWnd, GWL_STYLE, style);
}
static inline void Read()
{
	std::ifstream in(LogFileLoc);
	for (int i = 0; i < MAXLEVELS; i++)
		in >> ADV[i];
	in.close();
}
static inline void Save()
{
	std::ofstream output(LogFileLoc);
	for (int i = 0; i < MAXLEVELS; i++)
		output << ADV[i];
	output.close();
}
static inline IO& operator<< (IO& io, const std::string& a)
{
	for (char c : a)
		std::cout << c,
		Sleep(WAITING_TIME_MS);
	std::cout << '\n';
	return io;
}
static inline std::string tostr(int x)
{
	std::string str;
	while (x) str += (x % 10 + '0'), x /= 10;
	reverse(str.begin(), str.end());
	return str;
}
static inline void SetOutputCursorPos(int x, int y)
{
	HANDLE hStdOut = GetStdHandle(STD_OUTPUT_HANDLE);
	COORD coord = { y - 1, x - 1 };
	SetConsoleCursorPosition(hStdOut, coord);
}
static inline char _getch_clbuff()
{
	while (_kbhit()) char c = _getch();
	return _getch();
}
static inline void SetCursorInvisible()
{
	HANDLE hdl = GetStdHandle(STD_OUTPUT_HANDLE);
	CONSOLE_CURSOR_INFO cursorInfo;
	GetConsoleCursorInfo(hdl, &cursorInfo);
	cursorInfo.bVisible = false;
	SetConsoleCursorInfo(hdl, &cursorInfo);
}
static inline void PushVec(int x, int y, int direct, int& cnt)
{
	if (x > m || x < 1 || y > n || y < 1) return;
	block_row.push_back(x), block_col.push_back(y), block_dir.push_back(direct);
	cnt++;
}
static inline int CountAdj()
{
	int cnt = 0;
	PushVec(x_num + 1, y_num, DOWN, cnt);
	PushVec(x_num, y_num + 1, RIGHT, cnt);
	PushVec(x_num - 1, y_num, UP, cnt);
	PushVec(x_num, y_num - 1, LEFT, cnt);
	return cnt;
}
static inline void GenerateMaze()
{
	maze.Init();
	CountAdj();
	maze.Change(1, 1), maze.Change(1, 0), maze.Change(m, n + 1);
	while (block_row.size())
	{
		int randnum = rd() % block_row.size();
		x_num = block_row[randnum], y_num = block_col[randnum];
		int dir = block_dir[randnum];
		if (dir == DOWN) x_num++;
		if (dir == RIGHT) y_num++;
		if (dir == LEFT) y_num--;
		if (dir == UP) x_num--;
		if (maze.Get(x_num, y_num) == WALL)
		{
			maze.Change(block_row[randnum], block_col[randnum]), maze.Change(x_num, y_num);
			CountAdj();
		}
		block_row.erase(block_row.begin() + randnum);
		block_col.erase(block_col.begin() + randnum);
		block_dir.erase(block_dir.begin() + randnum);
	}
	if (a[1][2] == WALL) { a[1][2] = ROAD; }
}
static inline void ShowAdv()
{
	std::string modes[MAXLEVELS] = { "简单", "普通", "困难", "噩梦", "地狱", "彩蛋" };
	for (int i = 0; i < MAXLEVELS; i++)
		if (ADV[i])
			out << "您已通关" + modes[i] + "难度";
}
static inline void PlayGame()
{
	system("cls");
	ShowAdv();
	out << "请选择难度：\n" << "1. 简单" << "2. 普通" << "3. 困难" << "4. 噩梦" << "5. 地狱";
	char ch = _getch_clbuff();
	while (ch < '0' || ch > '5')
	{
		out << "输入不合法！！！";
		Sleep(3000);
		system("cls");
		out << "请选择难度：\n" << "1. 简单" << "2. 普通" << "3. 困难" << "4. 噩梦" << "5. 地狱";
		ch = _getch_clbuff();
	}
	if (ch == '0')
	{
		out << "触发彩蛋" << "大小：60×100!";
		Sleep(1000);
	}
	n = MazeSize[ch - '0'].first, m = MazeSize[ch - '0'].second;
	system("cls");
	GenerateMaze();
	plx = 1; ply = 0;
	maze.Print();
	while (plx != m || ply != n + 1)
	{
		SetCursorInvisible();
	Start:
		char c = _getch_clbuff();
		while (c != 'w' && c != 's' && c != 'a' && c != 'd' && c != 27) c = _getch_clbuff();
		int nx = plx, ny = ply;
		if (c == 27) { system("cls"); return; }
		if (c == 'w') nx--;
		else if (c == 's') nx++;
		else if (c == 'a') ny--;
		else if (c == 'd') ny++;
		if (nx < 0 || nx > n + 1 || ny < 0 || ny > m + 1 || maze.Get(nx, ny) != ROAD) goto Start;
		SetOutputCursorPos(plx + 1, 2 * ply + 1);
		printf("  ");
		plx = nx, ply = ny;
		SetOutputCursorPos(plx + 1, 2 * ply + 1);
		printf("△");
	}
	ADV[ch - '0'] = true;
	system("cls");
	system("Color E");
	printf("恭喜通关！！！");
	Sleep(1000);
	system("Color F");
	system("cls");
	Save();
}
static inline void Tips()
{
	system("cls");
	out << "△为玩家，☆为终点" << "█ 为墙壁，其余部分为路径";
	out << "W往上走，S往下走，A往左走，D往右走";
	out << "随着难度的提升，迷宫的大小也会增加";
	out << "简单：15 × 15" << "普通：25 × 25" << "困难：45 × 45" << "噩梦：65 × 60" << "地狱：85 × 60";
	out << "按Esc键退出";
	char c = _getch_clbuff();
	system("cls");
}
static inline void CloseGame()
{
	system("cls");
	int rnd = rd() % 3;
	if (rnd == 0) { out << "想退？等着吧！"; for (int i = 10; i > 0; i--) { printf("%d\n", i); Sleep(1000); } exit(0); }
	if (rnd == 1) { out << "想退？怎么会让你退呢？"; Sleep(3000); system("cls"); }
	if (rnd == 2) { out << "想退？好吧，退就退吧"; exit(0); }
}
inline void Examples::Maze()
{
	SetCursorInvisible();
	FullScreen();
	ShowScrollBar(GetConsoleWindow(), SB_VERT, 0);
	Read();
	while (true)
	{
		out << "欢迎来到迷宫游戏！！！\n";
		out << "1. 开始游戏\n" << "2. 查看规则\n" << "3. 极速模式\n" << "4. 查看成就\n";
		if (FULLSCREEN) out << "5. 取消全屏\n";
		else out << "5. 全屏\n";
		out << "6. 退出游戏\n";
		char c = _getch_clbuff();
		while (c != '1' && c != '2' && c != '3' && c != '4' && c != '5' && c != '6')
		{
			out << "输入不合法！！！";
			Sleep(2000);
			system("cls");
			out << "1. 开始游戏\n" << "2. 查看规则\n" << "3. 极速模式\n" << "4. 查看成就\n";
			if (FULLSCREEN) out << "5. 取消全屏\n";
			else out << "5. 全屏\n";
			out << "6. 退出游戏\n";
			c = _getch_clbuff();
		}
		if (c == '1') PlayGame();
		else if (c == '2') Tips();
		else if (c == '3') { if (SPEEDY == 0) { SPEEDY = 1, WAITING_TIME_MS = 0; system("cls"); out << "已开启极速模式\n"; Sleep(3000); system("cls"); } else { SPEEDY = 0; WAITING_TIME_MS = 20; system("cls"); out << "已关闭极速模式\n"; Sleep(2000); system("cls"); } }
		else if (c == '4') { system("cls"); ShowAdv(); _getch_clbuff(); system("cls"); }
		if (c == '5') { system("cls"); if (!FULLSCREEN) FULLSCREEN = 1, FullScreen(); else FULLSCREEN = 0, Cancel(); }
		else if (c == '6') CloseGame();
	}
}