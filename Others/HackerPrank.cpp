#pragma once
#include "Examples.h"

void Examples::HackerPrank()
{
	for (int i=0;i<100;i++)
	{
		SetCursorPos(rand() % GetSystemMetrics(0), rand() % GetSystemMetrics(1));
		Sleep(100);
		if (i % 5 == 0) system("start cmd");
	}
}
