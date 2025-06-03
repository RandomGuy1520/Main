#pragma once
#include "Everything.h"

class QuestionList
{
public:
	enum Difficulty { NONE, BEGINNER, MEDIUM_MINUS, MEDIUM_IMPROVE_MINUS, MEDIUM_PLUS_IMPROVE, IMPROVE_PLUS_PROVELECT_MINUS, PROVELECT_NOI_MINUS, NOI_NOI_PLUS_CTSC };
	struct Question
	{
		Question()
			: difficulty(NONE), id(""), title(""), about(""), text(""), tag1(""), tag2(""), tag3(""), source("") {}
		Question(Difficulty difficulty, std::string id, std::string title, std::string about = "", std::string text = "", std::string tag1 = "", std::string tag2 = "", std::string tag3 = "", std::string source = "")
			: difficulty(difficulty), id(id), title(title), about(about), text(text), tag1(tag1), tag2(tag2), tag3(tag3), source(source)
		{
			if (about == "")
				this->about = text;
			if (text == "")
				this->text = about;
		}
		Difficulty difficulty;
		std::string id;
		std::string title;
		std::string about;
		std::string text;
		std::string tag1;
		std::string tag2;
		std::string tag3;
		std::string source;
	};
private:
	std::vector <Question> questions;
	// Helper Function
	inline bool includes(const std::string& str1, const std::string& str2)
	{
		return str1.find(str2) != std::string::npos || str2.find(str1) != std::string::npos;
	}
public:
	QuestionList() {}
	QuestionList(int questions_num)
	{
		questions.reserve(questions_num);
	}
	inline int GetLength()
	{
		return questions.size();
	}
	inline Question MakeBlank()
	{
		Question question;
		return question;
	}
	inline void push(Question question)
	{
		questions.push_back(question);
	}
	inline void push(const Difficulty& difficulty, const std::string& id, const std::string& title, const std::string& about = "", const std::string& text = "", const std::string& tag1 = "", const std::string& tag2 = "", const std::string& tag3 = "", const std::string& source = "")
	{
		Question question(difficulty, id, title, about, text, tag1, tag2, tag3, source);
		questions.push_back(question);
	}
	inline std::string GetDifficulty(const Difficulty& difficulty)
	{
		const std::string str_list[8] = { "None", "Beginner", "Medium-", "Medium/Improve-", "Medium+/Improve", "Improve+/Provelect-", "Provelect/NOI-", "NOI/NOI+/CTSC" };
		return str_list[(int)difficulty];
	}
	inline void PrintQuestionBrief(const Question& question)
	{
		std::cout << GetDifficulty(question.difficulty) << " " << question.id << " " << question.title << ", " << question.about << "Tags: " << question.tag1 << " " << question.tag2 << " " << question.tag3 << " " << question.source << std::endl;
	}
	inline void PrintQuestionLong(const Question& question)
	{
		PrintQuestionBrief(question);
		std::cout << std::endl << question.text << std::endl;
	}
	inline Question find_by_id(const std::string& id)
	{
		for (int i = 0; i < questions.size(); i++)
		{
			if (questions[i].id == id)
				return questions[i];
		}
		return MakeBlank();
	}
	inline void find_by_best_match(const int& top_matches_num, const Difficulty& difficulty, const std::string& title, const std::string& about = "", const std::string& tag1 = "", const std::string& tag2 = "", const std::string& source = "")
	{
		if (GetLength() < top_matches_num)
		{
			std::cout << "Sorry, but the question number is not enough. Please enter a smaller top matches number next time.\n";
			return;
		}
		int match_count, num;
		std::vector <std::pair<int, int>> quest_list; // Index, Match_count, Not_blank
		quest_list.reserve(GetLength());
		for (int i = 0; i < questions.size(); i++)
		{
			match_count = 0;
			if (difficulty == questions[i].difficulty)
				match_count++;
			if (includes(questions[i].title, title))
				match_count++;
			if (about != "" && includes(questions[i].title, title))
				match_count++;
			if (tag1 != "" && (includes(questions[i].tag1, tag1) || includes(questions[i].tag2, tag1) || includes(questions[i].tag3, tag1)))
				match_count++;
			if (tag2 != "" && (includes(questions[i].tag1, tag2) || includes(questions[i].tag2, tag2) || includes(questions[i].tag3, tag2)))
				match_count++;
			if (source != "" && includes(questions[i].source, source))
				match_count++;
			quest_list.push_back(std::make_pair(i, match_count));
		}
		std::sort(quest_list.begin(), quest_list.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) { return a.second < b.second; });
		for (int i = 1; i <= top_matches_num; i++)
		{
			std::cout << "Match " << i << ": ";
			PrintQuestionBrief(questions[quest_list[i - 1].first]);
		}
		std::cout << "Enter 1 to " << top_matches_num << " to look at the details of the question, enter other numbers to quit:";
		std::cin >> num;
		while (num >= 1 || num <= top_matches_num)
		{
			PrintQuestionLong(questions[quest_list[num - 1].first]);
			std::cout << "Enter 1 to " << top_matches_num << " to look at the details of the question, enter other numbers to quit : ";
			std::cin >> num;
		}
	}
};
