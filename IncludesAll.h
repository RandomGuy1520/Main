// This IncludesAll.h is for TestBlank.cpp and other cpp files.

#pragma once
#include "Main.cpp"
#include "Others.h"
#include "BigNumbers.cpp"
#include "IO.cpp"
#include "OI.cpp"
#include "Luogu.cpp"
#include "Macros.h"

#define USING_NAMESPACE
#ifdef USING_NAMESPACE
using namespace std;
using namespace BigInt;
using namespace IO;
using namespace OI;
using namespace Main;
//using namespace Examples;
#endif
