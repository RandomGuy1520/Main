//#include "Main.cpp"
//using namespace std;
//
//template <typename T>
//class SparseTable
//{
//private:
//	const int n;
//	int maxst;
//	std::vector<T> orig;
//	std::vector<std::vector<T>> st;
//	T(*func)(T, T);
//	bool start_from_one = true;
//	void init()
//	{
//		for (int i = 0; i < n; i++) st[i][0] = orig[i];
//		for (int p = 1, j = 1; j <= maxst; p *= 2, j++)
//			for (int i = 0; i + p < n; i++)
//				st[i][j] = func(st[i][j - 1], st[i + p][j - 1]);
//	}
//public:
//	struct Functions
//	{
//		T min_func(T a, T b)
//		{
//			return min(a, b);
//		}
//		T max_func(T a, T b)
//		{
//			return max(a, b);
//		}
//		T gcd_func(T a, T b)
//		{
//			return std::gcd(a, b);
//		}
//		T lcm_func(T a, T b)
//		{
//			return std::lcm(a, b);
//		}
//	};
//	SparseTable(T* begin, T* end, T(*func)(T, T)) : func(func), n(end - begin)
//	{
//		maxst = ceil(log2(n));
//		for (T* i = begin; i <= end; i++) orig.push_back(*i);
//		std::vector<std::vector<T>> _st(n, std::vector<T>(maxst + 1, 0));
//		st = _st;
//		init();
//	}
//	SparseTable(const std::vector<T>& orig, T(*func)(T, T)) : func(func), n(orig.size()), orig(orig)
//	{
//		maxst = ceil(log2(n));
//		std::vector<std::vector<T>> _st(n, std::vector<T>(maxst + 1, 0));
//		st = _st;
//		init();
//	}
//	void SetStartFromZero()
//	{
//		start_from_one = false;
//	}
//	T query(const int& l, const int& r) const
//	{
//		int dist = floor(log2(r - l + 1));
//		if (start_from_one) return func(st[l - 1][dist], st[r - (1 << dist)][dist]);
//		else return func(st[l][dist], st[r - (1 << dist) + 1][dist]);
//	}
//};
//class JointSet
//{
//private:
//	const int n;
//	std::vector<int> fa;
//public:
//	JointSet(int n) : n(n)
//	{
//		fa = std::vector<int>(n + 1);
//		for (int i = 1; i <= n; i++) fa[i] = i;
//	}
//	int find(int idx)
//	{
//		return fa[idx] == idx ? idx : find(fa[idx]);
//	}
//	void merge(int x, int y)
//	{
//		fa[find(x)] = find(y);
//	}
//	bool is_connected(int x, int y)
//	{
//		return find(x) == find(y);
//	}
//};
//class WeightedJointSet
//{
//private:
//	const int n;
//	std::vector<std::pair<int, int> > fa;
//public:
//	WeightedJointSet(int n) : n(n)
//	{
//		fa = std::vector<std::pair<int, int> >(n + 1);
//		for (int i = 1; i <= n; i++) fa[i].first = i, fa[i].second = 0;
//	}
//	int find(int idx)
//	{
//		return fa[idx].first == idx ? idx : find(fa[idx].first);
//	}
//	void merge(int x, int y, int weight)
//	{
//		fa[find(x)].first = find(y);
//		fa[find(x)].second = weight;
//	}
//	bool is_connected(int x, int y)
//	{
//		return find(x) == find(y);
//	}
//};
//template <typename T>
//class MonotonicStack // Calls a stack, but really a std::vector
//{
//private:
//	bool (*func)(T, T); // From Up to Down of stack
//	std::vector<T> stack;
//public:
//	MonotonicStack(bool (*func)(T, T)) : func(func) {}
//	void push_back(const T& val)
//	{
//		while (!stack.empty() && func(stack.back(), val)) stack.pop_back();
//		stack.push_back(val);
//	}
//	void pop_back()
//	{
//		stack.pop_back();
//	}
//	T back()
//	{
//		return stack.back();
//	}
//	T operator[] (const int& idx) const
//	{
//		return stack[idx];
//	}
//	std::vector<T> GetStack()
//	{
//		return stack;
//	}
//};
//template <typename T>
//class MonotonicDeque
//{
//private:
//	int cnt = 0;
//	std::deque<T> q, idx;
//	const bool has_idx = false;
//	bool (*func)(T, T); // From Back to Front of deque
//public:
//	MonotonicDeque(bool (*func)(T, T)) : func(func) {}
//	MonotonicDeque(bool (*func)(T, T), bool has_idx) : func(func), has_idx(has_idx) {}
//	void push_back(const T& num)
//	{
//		while (!q.empty() && func(q.back(), num))
//		{
//			q.pop_back();
//			if (has_idx) idx.pop_back();
//		}
//		q.push_back(num);
//		if (has_idx) idx.push_back(++cnt);
//	}
//	void pop_back()
//	{
//		q.pop_back();
//	}
//	void push_front(const T& num)
//	{
//		q.push_front(num);
//	}
//	void pop_front()
//	{
//		q.pop_front();
//	}
//	void pop_front_until(int min_idx)
//	{
//		if (!has_idx) return;
//		while (idx.front() <= min_idx) idx.pop_front(), q.pop_front();
//	}
//};
////template <typename T>
////class SegmentTree
////{
////private:
////	const int n;
////	vector<T> segtree, orig;
////public:
////	SegmentTree(T* begin, T* end) : n(end - begin)
////	{
////		segtree = vector<T>(n << 2);
////		for (T* i = begin; i <= end; i++)
////			orig.push_back(*i);
////		build();
////	}
////	SegmentTree(vector<T> a) : n(a.size()), orig(a)
////	{
////		segtree = vector<T>(n << 2);
////		build();
////	}
////	void build(int l = -1, int r = -1, int idx = 1)
////	{
////		if (l == -1 && r == -1) l = 0, r = n - 1;
////		if (l == r)
////		{
////			segtree[idx] = orig[l];
////			return;
////		}
////		int m = l + ((r - l) >> 1);
////		build(l, m, idx * 2), build(m + 1, r, idx * 2 + 1);
////		segtree[idx] = segtree[idx * 2] + segtree[(idx * 2) + 1];
////	}
////	int getsum(int l, int r, int idx, int l_, int r_) {
////		if (l_ > r || r_ < l) return 0;
////		if (l <= l_ && r_ <= r) return segtree[idx];
////		pushdown(l_, r_, idx);
////		int mid = l_ + r_ >> 1;
////		return getsum(l, r, idx * 2, l_, mid) + getsum(l, r, idx * 2 + 1, mid + 1, r_);
////	}
////	void change(int l, int r, int idx, int val, int l_, int r_) {
////		if (l_ > r || r_ < l) return;
////		if (l <= l_ && r_ <= r) {
////			segtree[idx] += (r_ - l_ + 1) * val, lazy[idx] += val;
////			return;
////		}
////		pushdown(l_, r_, idx);
////		int mid = l_ + r_ >> 1;
////		change(l, r, idx * 2, val, l_, mid);
////		change(l, r, idx * 2 + 1, val, mid + 1, r_);
////		segtree[idx] = segtree[idx * 2] + segtree[idx * 2 + 1];
////	}
////	void change_func(T (*func)(T), int l, int r, int x = -1, int y = -1, int tree_idx = 1)
////	{
////		if (x == -1 && y == -1) x = 0, y = n - 1;
////		if (l > y || r < x) return;
////		if (l >= x && r <= y)
////		{
////			segtree[tree_idx] = func(segtree[tree_idx]);
////			return;
////		}
////		int mid = l + ((r - l) >> 1);
////		change_func(func, l, mid, x, y, 2 * tree_idx);
////		change_func(func, mid + 1, r, x, y, 2 * tree_idx + 1);
////	}
////	T getfunc(int l, int r, int idx, int l_, int r_) {
////		if (l_ > r || r_ < l) return 0;
////		if (l <= l_ && r_ <= r) return segtree[idx];
////		pushdown(l_, r_, idx);
////		int mid = l_ + r_ >> 1;
////		return getsum(l, r, idx * 2, l_, mid) + getsum(l, r, idx * 2 + 1, mid + 1, r_);
////	}
////};
//template <typename T>
//class TreelikeArray
//{
//private:
//	const int n;
//	std::vector<int> a;
//public:
//	TreelikeArray(int n) : n(n)
//	{
//		a = std::vector<int>(n + 1);
//	}
//	int lowbit(int x)
//	{
//		return x & -x;
//	}
//	int getsum(int x)
//	{
//		int cnt = 0;
//		while (x)
//		{
//			cnt = cnt + a[x];
//			x = x - lowbit(x);
//		}
//		return cnt;
//	}
//	int getsum(int l, int r)
//	{
//		return getsum(r) - getsum(l - 1);
//	}
//	void add(int x, int k)
//	{
//		while (x <= n)
//		{
//			a[x] += k;
//			x = x + lowbit(x);
//		}
//	}
//};
//
//class BinaryTree
//{
//private:
//	struct node {
//		int l, r, fa;
//	};
//public:
//};
//
//template <typename T>
//class Tree
//{
//private:
//	const int n;
//	int root, tree_stmax;
//	bool has_val = false, is_init_tree_st = false;
//	struct node
//	{
//		int fa = -1, depth = -1;
//		T val = 0;
//		std::vector<int> to;
//		node() {}
//		node(int fa) : fa(fa) {}
//		node(std::vector<int> to) : to(to) {}
//		node(std::vector<int> to, T val) : val(val), to(to) {}
//		node(int fa, std::vector<int> to) : fa(fa), to(to) {}
//		node(int fa, T val, std::vector<int> to) : fa(fa), val(val), to(to) {}
//	};
//	std::vector<std::vector<int> > tree_st;
//public:
//	std::vector<node> tree;
//	Tree(int n) : n(n), root(1), tree_stmax(ceil(log2(n))) {}
//	Tree(int n, int root) : n(n), root(root), tree_stmax(ceil(log2(n))) {}
//	Tree(int n, std::vector<int> to[]) : n(n), root(1), tree_stmax(ceil(log2(n)))
//	{
//		tree.push_back(node());
//		for (int i = 1; i <= n; i++) tree.push_back(node(to[i]));
//		init(root);
//	}
//	Tree(std::vector<int>* to_begin, std::vector<int>* to_end) : n(to_end - to_begin), root(1)
//	{
//		tree_stmax = ceil(log2(n));
//		for (std::vector<int>* i = to_begin; i <= to_end; i++) tree.push_back(node(*i));
//		init(root);
//	}
//	Tree(int n, std::vector<int> to[], int root) : n(n), root(root), tree_stmax(ceil(log2(n)))
//	{
//		tree.push_back(node());
//		for (int i = 1; i <= n; i++) tree.push_back(node(to[i]));
//		init(root);
//	}
//	Tree(std::vector<int>* to_begin, std::vector<int>* to_end, int root) : n(to_end - to_begin), root(root)
//	{
//		tree_stmax = ceil(log2(n));
//		for (std::vector<int>* i = to_begin; i <= to_end; i++) tree.push_back(node(*i));
//		init(root);
//	}
//	Tree(int n, std::pair<T, std::vector<int> > to[]) : has_val(true), n(n), root(1), tree_stmax(ceil(log2(n)))
//	{
//		tree.push_back(node());
//		for (int i = 1; i <= n; i++) tree.push_back(node(to[i].second, to[i].first));
//		init(root);
//	}
//	Tree(std::pair<T, std::vector<int> >* to_begin, std::pair<T, std::vector<int> >* to_end) : has_val(true), n(to_end - to_begin), root(1)
//	{
//		tree_stmax = ceil(log2(n));
//		tree.push_back(node());
//		for (std::pair<T, std::vector<int> >* i = to_begin; i <= to_end; i++) tree.push_back(node((*i).second, (*i).first));
//		init(root);
//	}
//	Tree(int n, std::pair<T, std::vector<int> > to[], int root) : has_val(true), n(n), root(root), tree_stmax(ceil(log2(n)))
//	{
//		tree.push_back(node());
//		for (int i = 1; i <= n; i++) tree.push_back(node(to[i].second, to[i].first));
//		init(root);
//	}
//	Tree(std::pair<T, std::vector<int> >* to_begin, std::pair<T, std::vector<int> >* to_end, int root) : has_val(true), n(to_end - to_begin), root(root)
//	{
//		tree_stmax = ceil(log2(n));
//		tree.push_back(node());
//		for (std::pair<T, std::vector<int> >* i = to_begin; i <= to_end; i++) tree.push_back(node((*i).second, (*i).first));
//		init(root);
//	}
//	void init(int now, int father = -1)
//	{
//		if (!has_val) tree[now].val = now;
//		tree[now].depth = tree[father].depth + 1;
//		tree[now].fa = father;
//		for (int& i : tree[now].to)
//			if (i != father)
//				init(i, now);
//	}
//	void init_tree_st()
//	{
//		tree_st = std::vector<std::vector<int> >(n, std::vector<int>(tree_stmax + 1, 0));
//		for (int i = 1; i <= n; i++) tree_st[i][0] = tree[i].fa;
//		for (int j = 1; j <= tree_stmax; j++)
//			for (int i = 1; i <= n; i++)
//				tree_st[i][j] = tree_st[tree_st[i][j - 1]][j - 1];
//	}
//	void insert(int l, int r, int length)
//	{
//		tree[l].to.push_back(r);
//		tree[r].to.push_back(l);
//	}
//	int GetRoot()
//	{
//		return root;
//	}
//	int ChangeRoot(int new_root)
//	{
//		root = new_root;
//		init(new_root);
//	}
//	int GetNodeIdx(T val)
//	{
//		for (int i = 0; i < tree.size(); i++)
//			if (tree[i].val == val)
//				return i;
//	}
//	void dfs(void* func(node a), int start)
//	{
//		func(tree[start]);
//		for (int& i : tree[start].to)
//			if (i != tree[start].fa)
//				dfs(func, i);
//	}
//	int dfscnt(int func(node a), int start)
//	{
//		int cnt = func(tree[start]);
//		for (int& i : tree[start].to)
//			if (i != tree[start].fa)
//				cnt += dfscnt(func, i);
//		return cnt;
//	}
//	void bfs(void* func(node a), int start)
//	{
//		std::queue<int> idx_queue;
//		idx_queue.push(start);
//		while (!idx_queue.empty())
//		{
//			int idx = idx_queue.front();
//			func(tree[idx]);
//			for (int& i : tree[idx].to)
//				if (i != tree[idx].fa)
//					idx_queue.push(i);
//			idx_queue.pop();
//		}
//	}
//	int bfscnt(int func(node a), int start)
//	{
//		int cnt = 0;
//		std::queue<int> idx_queue;
//		idx_queue.push(start);
//		while (!idx_queue.empty())
//		{
//			int idx = idx_queue.front();
//			cnt += func(tree[idx]);
//			for (int& i : tree[idx].to)
//				if (i != tree[idx].fa)
//					idx_queue.push(i);
//			idx_queue.pop();
//		}
//		return cnt;
//	}
//	int lca(int l, int r)
//	{
//		if (!is_init_tree_st) init_tree_st(), is_init_tree_st = true;
//		int _l, _r;
//		if (has_val) _l = GetNodeIdx(l), _r = GetNodeIdx(r);
//		else _l = l, _r = r;
//		if (tree[_l].depth < tree[_r].depth) std::swap(_l, _r);
//		int diff = tree[_l].depth - tree[_r].depth;
//		for (int i = tree_stmax; i >= 0; i--) if ((1 << i) & diff) _l = tree_st[_l][i];
//		for (int i = tree_stmax; i >= 0; i--) if (tree_st[_l][i] != tree_st[_r][i]) _l = tree_st[_l][i], _r = tree_st[_r][i];
//		return (_l == _r ? _l : tree[_l].fa);
//	}
//};
//template <typename T>
//class WeightedTree
//{
//private:
//	const int n;
//	int root, tree_stmax;
//	bool has_val = false, is_init_tree_st = false;
//	struct node
//	{
//		int fa = -1, depth = -1, segment_depth = -1;
//		T val = 0;
//		std::vector<std::pair<int, int> > to; // std::vector<<dest_idx, length>>
//		node() {}
//		node(int fa) : fa(fa) {}
//		node(std::vector<std::pair<int, int> > to) : to(to) {}
//		node(std::vector<std::pair<int, int> > to, T val) : val(val), to(to) {}
//		node(int fa, std::vector<std::pair<int, int> > to) : fa(fa), to(to) {}
//		node(int fa, T val, std::vector<std::pair<int, int> > to) : fa(fa), val(val), to(to) {}
//	};
//	std::vector<std::vector<int> > tree_st;
//public:
//	std::vector<node> tree;
//	WeightedTree(int n) : n(n), root(1), tree_stmax(ceil(log2(n))) {}
//	WeightedTree(int n, int root) : n(n), root(root), tree_stmax(ceil(log2(n))) {}
//	WeightedTree(int n, std::vector<std::pair<int, int> > to[]) : n(n), root(1), tree_stmax(ceil(log2(n)))
//	{
//		tree.push_back(node());
//		for (int i = 1; i <= n; i++) tree.push_back(node(to[i]));
//		init(root);
//	}
//	WeightedTree(std::vector<std::pair<int, int> >* to_begin, std::vector<std::pair<int, int> >* to_end) : n(to_end - to_begin), root(1)
//	{
//		tree_stmax = ceil(log2(n));
//		for (std::vector<std::pair<int, int> >* i = to_begin; i <= to_end; i++) tree.push_back(node(*i));
//		init(root);
//	}
//	WeightedTree(int n, std::vector<std::pair<int, int> > to[], int root) : n(n), root(root), tree_stmax(ceil(log2(n)))
//	{
//		tree.push_back(node());
//		for (int i = 1; i <= n; i++) tree.push_back(node(to[i]));
//		init(root);
//	}
//	WeightedTree(std::vector<std::pair<int, int> >* to_begin, std::vector<std::pair<int, int> >* to_end, int root) : n(to_end - to_begin), root(root)
//	{
//		tree_stmax = ceil(log2(n));
//		for (std::vector<std::pair<int, int> >* i = to_begin; i <= to_end; i++) tree.push_back(node(*i));
//		init(root);
//	}
//	WeightedTree(int n, std::pair<T, std::vector<std::pair<int, int> > > to[]) : has_val(true), n(n), root(1), tree_stmax(ceil(log2(n)))
//	{
//		tree.push_back(node());
//		for (int i = 1; i <= n; i++) tree.push_back(node(to[i].second, to[i].first));
//		init(root);
//	}
//	WeightedTree(std::pair<T, std::vector<std::pair<int, int> > >* to_begin, std::pair<T, std::vector<std::pair<int, int> > >* to_end) : has_val(true), n(to_end - to_begin), root(1)
//	{
//		tree_stmax = ceil(log2(n));
//		tree.push_back(node());
//		for (std::pair<T, std::vector<std::pair<int, int> > >* i = to_begin; i <= to_end; i++) tree.push_back(node((*i).second, (*i).first));
//		init(root);
//	}
//	WeightedTree(int n, std::pair<T, std::vector<std::pair<int, int> > > to[], int root) : has_val(true), n(n), root(root), tree_stmax(ceil(log2(n)))
//	{
//		tree.push_back(node());
//		for (int i = 1; i <= n; i++) tree.push_back(node(to[i].second, to[i].first));
//		init(root);
//	}
//	WeightedTree(std::pair<T, std::vector<std::pair<int, int> > >* to_begin, std::pair<T, std::vector<std::pair<int, int> > >* to_end, int root) : has_val(true), n(to_end - to_begin), root(root)
//	{
//		tree_stmax = ceil(log2(n));
//		tree.push_back(node());
//		for (std::pair<T, std::vector<std::pair<int, int> > >* i = to_begin; i <= to_end; i++) tree.push_back(node((*i).second, (*i).first));
//		init(root);
//	}
//	void init(int now, int father = -1, int length = -1)
//	{
//		if (!has_val) tree[now].val = now;
//		tree[now].depth = tree[father].depth + 1;
//		tree[now].segment_depth = tree[father].segment_depth + length;
//		tree[now].fa = father;
//		for (std::pair<int, int>& i : tree[now].to)
//			if (i.first != father)
//				init(i.first, now, i.second);
//	}
//	void init_tree_st()
//	{
//		tree_st = std::vector<std::vector<int> >(n, std::vector<int>(tree_stmax + 1, 0));
//		for (int i = 1; i <= n; i++) tree_st[i][0] = tree[i].fa;
//		for (int j = 1; j <= tree_stmax; j++)
//			for (int i = 1; i <= n; i++)
//				tree_st[i][j] = tree_st[tree_st[i][j - 1]][j - 1];
//	}
//	void insert(int l, int r, int length)
//	{
//		tree[l].to.push_back(std::make_pair(r, length));
//		tree[r].to.push_back(std::make_pair(l, length));
//	}
//	int GetRoot()
//	{
//		return root;
//	}
//	int ChangeRoot(int new_root)
//	{
//		root = new_root;
//		init(new_root);
//	}
//	int GetNodeIdx(T val)
//	{
//		for (int i = 0; i < tree.size(); i++)
//			if (tree[i].val == val)
//				return i;
//	}
//	void dfs(void* func(node a), int start)
//	{
//		func(tree[start]);
//		for (std::pair<int, int>& i : tree[start].to)
//			if (i.first != tree[start].fa)
//				dfs(func, i.first);
//	}
//	int dfscnt(int func(node a), int start)
//	{
//		int cnt = func(tree[start]);
//		for (std::pair<int, int>& i : tree[start].to)
//			if (i.first != tree[start].fa)
//				cnt += dfscnt(func, i.first);
//		return cnt;
//	}
//	void bfs(void* func(node a), int start)
//	{
//		std::queue<int> idx_queue;
//		idx_queue.push(start);
//		while (!idx_queue.empty())
//		{
//			int idx = idx_queue.front();
//			func(tree[idx]);
//			for (std::pair<int, int>& i : tree[idx].to)
//				if (i.first != tree[idx].fa)
//					idx_queue.push(i.first);
//			idx_queue.pop();
//		}
//	}
//	int bfscnt(int func(node a), int start)
//	{
//		int cnt = 0;
//		std::queue<int> idx_queue;
//		idx_queue.push(start);
//		while (!idx_queue.empty())
//		{
//			int idx = idx_queue.front();
//			cnt += func(tree[idx]);
//			for (std::pair<int, int>& i : tree[idx].to)
//				if (i.first != tree[idx].fa)
//					idx_queue.push(i.first);
//			idx_queue.pop();
//		}
//		return cnt;
//	}
//	int lca(int l, int r)
//	{
//		if (!is_init_tree_st) init_tree_st(), is_init_tree_st = true;
//		int _l, _r;
//		if (has_val) _l = GetNodeIdx(l), _r = GetNodeIdx(r);
//		else _l = l, _r = r;
//		if (tree[_l].depth < tree[_r].depth) std::swap(_l, _r);
//		int diff = tree[_l].depth - tree[_r].depth;
//		for (int i = tree_stmax; i >= 0; i--) if ((1 << i) & diff) _l = tree_st[_l][i];
//		for (int i = tree_stmax; i >= 0; i--) if (tree_st[_l][i] != tree_st[_r][i]) _l = tree_st[_l][i], _r = tree_st[_r][i];
//		return (_l == _r ? _l : tree[_l].fa);
//	}
//};
//
//class Graph
//{
//private:
//public:
//	Graph() {}
//};
//
//class DirectedGraph
//{
//	struct node {
//		vector<int> fa, to;
//	};
//};
