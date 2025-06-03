#pragma once
#include "Examples.h"
static const int LEN = 5;

static std::vector<std::string> words, possible_words, tmp;

static inline std::string get(std::string str, std::string query)
{
	char c[LEN];
	std::string s;
	int num[30]{}, eq[30]{}; for (char c : str) num[c - 'a']++;
	for (int i = 0; i < query.size(); i++)
	{
		if (str[i] == query[i]) c[i] = 'A', eq[query[i] - 'a']++;
	}
	for (int i = 0; i < query.size(); i++)
	{
		if (str[i] == query[i]);
		else if (num[query[i] - 'a'] > 0 && eq[query[i] - 'a'] == 0) c[i] = 'B';
		else c[i] = 'C';
	}
	s = c;
	return s;
}

static inline std::string simple()
{
	int mn = 2147483647; std::string best;
	if (possible_words.size() == 1)
		return possible_words[0];
	for (std::string s : words)
	{
		std::unordered_map<std::string, int> mp;
		for (std::string t : possible_words)
			if (mp.find(get(s, t)) == mp.end())
				mp[get(s, t)] = 1;
			else mp[get(s, t)]++;
		int cnt = 0;
		for (std::pair<std::string, int> p : mp)
			cnt += p.second * p.second;
		if (mn > cnt || mn == cnt && mp.find("AAAAA") != mp.end())
			mn = cnt, best = s;
	}
	return best;
}

inline void Examples::Wordle()
{
	FILE* f = freopen("words.txt", "r", stdin);
	std::string s;
	while (std::cin >> s)
		if (s.size() == LEN)
			words.push_back(s);
	possible_words = words;
	fclose(f);
	freopen("con", "r", stdin);
	std::cout << "Initialization completed!\nWelcome to Wordle Helper!\n";
	std::string last_result;
	bool first_time = true;
	while (true)
	{
		std::string guess = (first_time ? "tares" : simple());
		std::cout << "Possible words num: " << possible_words.size() << std::endl;
		std::cout << "Preference guess: " << guess << std::endl;
		std::cout << "Enter results: " << std::endl;
		char inp[5];
		scanf("%s", inp);
		last_result = inp;
		std::cout << std::endl;
		if (last_result == "AAAAA")
		{
			std::cout << "Guessed it! What a human computer <ÈË»ú>!\n";
			return;
		}
		tmp = possible_words; possible_words.clear();
		for (std::string str : tmp)
			if (get(str, guess) == last_result)
				possible_words.push_back(str), std::cout << str << std::endl;
		first_time = false;
	}
}
