
# 1.动态规划

qzy整理

**题目特点**：

* [计数](#计数)

  * [1. 斐波那契数](#1-斐波那契数)  
  
  * [2. 爬楼梯](#2-爬楼梯)
  
  * [3. 组合总和 Ⅳ](#3-组合总和-iv)

  * [4. 不同路径](#4-不同路径)
  
  * [5. 解码方法](#5-解码方法)
  
  * [6. 不同路径 II](#6-不同路径-ii)

* [求最大最小值](#求最大最小值)
  
  * [1. 打家劫舍](#1-打家劫舍)
  
  * [2. 打家劫舍 II](#2-打家劫舍-ii)
  
  * [3. 使用最小花费爬楼梯](#3-使用最小花费爬楼梯)

  * [4. 最大子序和](#4-最大子序和)
  
  * [5. 最小路径和](#5-最小路径和)
  
  * [6. 三角形最小路径和](#6-三角形最小路径和)
  
  * [7. 买卖股票的最佳时机](#7-买卖股票的最佳时机)
  
  * [8. 整数拆分](#8-整数拆分)
  
  * [9. 最长递增子序列](#9-最长递增子序列)
  
  * [10. 最长连续递增序列](#10-最长连续递增序列)
  
  * [11. 最长重复子数组](#11-最长重复子数组)
  
  * [12. 最长公共子序列](#12-最长公共子序列)
  
  * [13. 不相交的线](#13-不相交的线)
  
* [求存在性](#求存在性)

**系列分类**：

* 打家劫舍系列：
  
  * [开始打家劫舍](#1-打家劫舍)

  * [再度打家劫舍](#2-打家劫舍-ii)

* 子序列
  
  * [最大子序和](#4-最大子序和)
  
  * [最长递增子序列](#9-最长递增子序列)
  
  * [最长连续递增序列](#10-最长连续递增序列)

  * [最长重复子数组](#11-最长重复子数组)
  
  * [最长公共子序列](#12-最长公共子序列)
  
  * [不相交的线](#13-不相交的线)

* 背包问题

需要明确背包问题的类型：

1. 0-1背包：背包体积，商品重量，商品价值

  * [分割等和子集]()

  
**解题步骤**：

1. 确定dp数组（dp table）以及下标的含义

2. 确定递推公式

3. dp数组如何初始化

4. 确定遍历顺序

5. 举例推导dp数组

## 计数

### 1. 斐波那契数

**tips**：此题进阶为[爬楼梯](#2-爬楼梯)

[Leetcode-509-Easy](https://leetcode-cn.com/problems/fibonacci-number/description/)

**题目描述**：**最简单的动规**，通常用 F(n) 表示，形成的序列称为斐波那契数列。该数列由 0 和 1 开始，后面的每一项数字都是前面两项数字的和。

**定义dp**：dp[i] 表示第i个数的斐波那契数值

**状态转移**：dp[i] = dp[i - 1] + dp[i - 2]

O(N) 时间和空间复杂度：

```python3
class Solution:
    def fib(self, N: int) -> int:
        dp = [0]*(N+1)
        if N <= 1:
            return N
        dp[0] = 0
        dp[1] = 1
        for i in range(2, N+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[-1]
```

O(N) 时间复杂度和 O(1) 空间复杂度：

```python3
class Solution:
    def fib(self, N: int) -> int:
        if N <= 1:
            return N
        pre = 0
        cur = 1
        for i in range(2, N+1):
            pre, cur = cur, pre + cur
        return cur
```

### 2. 爬楼梯

**tips**：此题可以改成求最值版本[使用最小花费爬楼梯](#3-使用最小花费爬楼梯)

[Leetcode-70-Easy](https://leetcode.com/problems/climbing-stairs/description/)

**题目描述**：需要 n 阶你才能到达楼顶。每次你可以爬1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢

**定义dp**：数组 dp 存储上楼梯的方法数，dp[i] 表示走到第 i 个楼梯的方法数目

**状态转移**：第 i 个楼梯可以由第 i-1 和 i-2 个楼梯再走 1 或 2 步到达。走到第 i 个楼梯的方法数为第 i-1 和第 i-2 个楼梯方法的总和。

O(N) 时间和空间复杂度：

```python3
class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 2:
            return n
        dp = [1]*(n+1)
        for i in range(2, n+1):
            dp[i] = dp[i-1]+dp[i-2]
        return dp[-1]
```

O(N) 时间复杂度和 O(1) 空间复杂度：

```python3
class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 2:
            return n
        cur = 1
        pre = 1
        for i in range(2, n+1):
            cur, pre = cur + pre, cur
        return cur
```

### 3. 组合总和 IV

[Leetcode-377-Medium](https://leetcode-cn.com/problems/combination-sum-iv/description/)

**题目描述**：一个由 不同 整数组成的数组 nums ，和一个目标整数 target 。请你从 nums 中找出并返回总和为 target 的元素组合的个数。

**定义dp**：数组 dp 存储组合个数，dp[i] 表示和为 i 的组合个数。

**状态转移**：dp[i] 可以看作由 dp[i - nums[j]] 构成，具体可以参考[树状图](https://leetcode-cn.com/problems/combination-sum-iv/solution/dong-tai-gui-hua-python-dai-ma-by-liweiwei1419/)。

```python3
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp = [0]*(target+1)
        dp[0] = 1
        for i in range(1, target+1):
            for j in nums:
                if i - j >= 0:
                    dp[i] += dp[i-j]
        return dp[-1]
```

### 4. 不同路径

[Leetcode-62-Medium](https://leetcode-cn.com/problems/unique-paths/description/)

**tips**：强烈建议此题结合[最小路径和](#4-最小路径和)一起学习。此题的进阶版为[不同路径 II](#6-不同路径-ii)。

**题目描述**：从 m x n 网格的左上角开始，每次只能向下或者向右移动一格，问有多少种方法可以到达右下角。

**定义dp**：数组 dp 存储路径方法数。dp[i][j] 为到达 (i, j) 格子处的方法数。

**状态转移**：走到第 (i, j) 格子处的情况只有两种，从它的上一格 (i, j-1) 往下一格或者从它的左边一格 (i-1, j) 往右一格。dp[i][j] =  dp[i - 1][j] + dp[i][j - 1]。特别的，第一行与第一列都只有一种方法到达，可以作为初始条件。

```python3
import numpy as np
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = np.ones(shape=(m, n))
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return int(dp[-1][-1])
```

### 5. 解码方法

[Leetcode-91-Medium](https://leetcode-cn.com/problems/decode-ways/description/)

**tips**：很像[爬楼梯](#2-爬楼梯)

**题目描述**：A-Z 的消息通过 "A":1, "B":2, ··· 映射进行了编码，给你一个只含数字的**非空**字符串 num ，请计算并返回解码方法的总数 。

**定义dp**：定义 dp 为解码总数， dp[i] 为到第 i 个数的解码总数。

**状态转移**：第 i 个位置的解码数只和两个位置有关：1. i 处单个字符解码，因而解码总数等于第 i-1 处位置的解码总数。2. i-1 与 i 位置组成两位数，映射到其对应的字母，因而解码总数等于第 i-2 处位置的解码总数。需要考虑哪些两位数能够映射成字母，很显然只有 10, 11, 12, ···, 25, 26 可以映射成字母。也需要考虑如果 i 处为0，那必然需要和前一个数结合才能构成合法的映射。

**ps**:此题边界要考虑清楚，剪枝条件较多，要考虑清楚。

O(N) 时间和空间复杂度：

```python3
class Solution:
    def numDecodings(self, s: str) -> int:
        dp = [0]*(len(s)+1)
        if s[0] == "0":
            return 0
        dp[0] = 1  # 无实际作用
        dp[1] = 1  # 第一个数若不为0，则第一个数映射方法为1，作为入口
        for i in range(2, len(s)+1):
            if s[i-1] == "0":
                if s[i-2] != "1" and s[i-2] != "2":
                    return 0
                else:
                    dp[i] = dp[i-2]
            else:
                if s[i - 2] == "1" or (s[i - 2] == "2" and "1" <= s[i-1] <= "6"):
                    dp[i] += dp[i-2]
                dp[i] += dp[i-1]
        return dp[-1]
```

O(N) 时间复杂度和 O(1) 空间复杂度：

```python3
```

### 6. 不同路径 II

[Leetcode-63-Medium](https://leetcode-cn.com/problems/unique-paths-ii/description/)

**题目描述**：从 m x n 网格的左上角开始，每次只能向下或者向右移动一格，现在考虑网格中有障碍物。问有多少种方法可以到达右下角。

**定义dp**：dp[i][j] 为到达 (i, j) 格子处的方法数。

**状态转移**：见[不同路径](#4-不同路径)。路障所处位置无法到达，dp[i][j] 应该为 0 。初始化的时候也要注意。

```python3
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        if obstacleGrid[0][0] == 1:
            return 0
        m = len(obstacleGrid)
        n =len(obstacleGrid[0])
        dp = [[0 for i in range(n)] for j in range(m)]
        for i in range(m):
            if obstacleGrid[i][0] == 0:
                dp[i][0] = 1
            else:
                break
        for i in range(n):
            if obstacleGrid[0][i] == 0:
                dp[0][i] = 1
            else:
                break
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 1:
                    continue              
                dp[i][j] = dp[i-1][j]+dp[i][j-1] 
        return dp[-1][-1]
```

## 求最大最小值

### 1. 打家劫舍

[Leetcode-198-Easy](https://leetcode.com/problems/house-robber/description/)

**题目描述**：抢劫一排住户，但是不能抢邻近的住户，求最大抢劫量。给定一个数组代表每个房屋存放金额。

**定义dp**：数组 dp 存储最大抢劫钱数，dp[i] 表示抢到第 i 家时的最大钱数。

**状态转移**：由于不能抢劫相邻的两家，所以抢到第 i 家的最大钱数只有两种可能：1.到第 i-2 家抢的钱数加上第 i 家的钱数。2.到第 i-1 家抢的钱数。只需求取两者最大值即可。

O(N) 时间和空间复杂度：

```python3
class Solution:
    def rob(self, nums: List[int]) -> int:
        dp = [0]*(len(nums)+1)
        dp[1] = nums[0]
        for i in range(2, len(nums)+1):
            dp[i] = max(dp[i-2]+nums[i-1], dp[i-1])
        return dp[-1]
```

O(N) 时间复杂度和 O(1) 空间复杂度：

```python3
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        pre2 = 0
        pre1 = nums[0]
        cur = nums[0]
        for i in range(1, len(nums)):
            cur, pre2=max(pre2 + nums[i], pre1), pre1
            pre1 = cur
        return cur
```

### 2. 打家劫舍 II

[Leetcode-213-medium](https://leetcode.com/problems/house-robber-ii/description/)

**题目描述**：同上一题打家劫舍，只不过所有的房屋都围成一个圈

**定义dp**：数组 dp 存储最大抢劫钱数，dp[i] 表示抢到第 i 家时的最大钱数。

**状态转移**：同上一题。由于是一个环，因此如果偷了第一家则最后一家不会被偷，同样如果偷了最后一家则第一家不会被偷，所以可以进行两次遍历，一次不包含第一家，一次不包含最后一家。

O(N) 时间复杂度和 O(1) 空间复杂度：

```python3
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        def steal(nums):
            dp = [0]*(len(nums)+1)
            dp[1] = nums[0]
            for i in range(2, len(nums)+1):
                dp[i] = max(dp[i-2]+nums[i-1], dp[i-1])
            return dp[-1]
        return max(steal(nums[:-1]), steal(nums[1:]))
```

### 3. 使用最小花费爬楼梯

[Leetcode-746-easy](https://leetcode-cn.com/problems/min-cost-climbing-stairs/description/)

**tips**：[爬楼梯](#2-爬楼梯)的进阶，但更[打家劫舍](#1-打家劫舍)

**题目描述**：一次爬两节或一节台阶，爬每层楼都消耗体力 cost[i]。求最小花费体力。

**定义dp**：dp[i] 表示爬到第 i 层花费的最少体力。

**状态转移**：dp[i] 由两个途径得到，一个是 dp[i-1] 一个是 dp[i-2]。 dp[i] = min(dp[i-1] + cost[i-1], dp[i-2] + cost(i-2))

O(N) 时间和空间复杂度：

```python3
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        dp = [0]*(len(cost)+1)
        for i in range(2, len(cost)+1):
            dp[i] = min(dp[i-1]+cost[i-1], dp[i-2]+cost[i-2])
        #print(dp)
        return dp[-1]
```

O(N) 时间复杂度和 O(1) 空间复杂度：

```python3
class Solution:
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        pre = 0
        cur = 0
        for i in range(2, len(cost)+1):
            cur, pre = min(cur+cost[i-1], pre+cost[i-2]), cur
        return cur
```

### 4. 最大子序和

[Leetcode-53-easy](https://leetcode-cn.com/problems/maximum-subarray/description/)

**题目描述**：给定一个整数数组 nums ，找到一个具有最大和的连续子数组，返回最大和

**定义dp**：定义 dp[i] 为以第 i 个数结尾的连续子数的最大和

**状态转移**：需要考虑第 i 个数是否应该加入前面的子序中，因此只用考虑两种可能：1.第 i 个数比前面子序的和更大。2.第 i 个数比到 i 的子序和更大。 dp[i] 更新成两者较大值即可。转移方程：dp[i] = max(dp[i-1]+nums[i], nums[i])

```python3
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        dp = [0]*(len(nums))
        dp[0] = nums[0]
        for i in range(1, len(nums)):
            dp[i] = max(dp[i-1]+nums[i], nums[i])
        return max(dp)
```

### 5. 最小路径和

[Leetcode-64-easy](https://leetcode-cn.com/problems/minimum-path-sum/description/)

**tips**：强烈建议此题结合[不同路径](#3-不同路径)一起学习。

**题目描述**：给定一个包含非负整数的 m x n 网格 grid ，每次只能向右或向左移动一格，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

**定义dp**：数组 dp 存储当前点的路径最小值。dp[i][j] 为到 (i, j) 格子处路径的最小值。

**状态转移**：走到第 (i, j) 格子处的情况只有两种，从它的上一格 (i, j-1) 往下一格或者从它的左边一格 (i-1, j) 往右一格。只需用左侧或者上面路径的最小值加上所在格子的数值即可。特别的，第一行和第一列的最短路径很容易求出。

```python3
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        dp = [[0 for i in range(n)] for j in range(m)]
        dp[0][0] = grid[0][0]
        for i in range(1, m):
            dp[i][0] = grid[i][0] + dp[i-1][0]
        for i in range(1, n):
            dp[0][i] = grid[0][i] + dp[0][i-1]
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
        return dp[-1][-1]
```

### 6. 三角形最小路径和

[Leetcode-120-easy](https://leetcode-cn.com/problems/triangle/description/)

**题目描述**：给定一个三角形 triangle ，找出自顶向下的最小路径和。每一步只能移动到下一行中相邻的结点上。

**定义dp**：数组 dp 存储当前点的路径最小值。dp[i][j] 为到 (i, j) 处路径的最小值。

**状态转移**：走到第 (i, j) 格子处的情况只有两种，从 (i-1, j-1) 或 (i-1, j) 到达。取最小值即可。题目要求找出自顶向下最小路径和，因此答案即为 dp 最后一行的最小值。

```python3
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        n = len(triangle)
        dp = [[0] * n for _ in range(n)]
        dp[0][0] = triangle[0][0]
        for i in range(1, n):
            dp[i][0] = dp[i-1][0] + triangle[i][0]
            dp[i][i] = dp[i-1][i-1] + triangle[i][i]
            for j in range(1, i):
                dp[i][j] = min(dp[i-1][j-1], dp[i-1][j]) + triangle[i][j]
        return min(dp[-1])
```

### 7. 买卖股票的最佳时机

[Leetcode-121-easy](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/description/)

**题目描述**：给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。你只能选择某一天买入这只股票，并选择在未来的某一个不同的日子卖出该股票。设计一个算法来计算你所能获取的最大利润。

**定义dp**：设置三个状态分别为 dp0，dp1，dp2。
    dp0：一直不买
    dp1：只买了一次
    dp2：买了一次，卖了一次
    初始化三种状态：
dp0 = 0
dp1 = - prices[0]
dp2 = float("-inf")

**状态转移**：
dp0 = 0：一直为0
dp1 = max(dp1, dp0 - prices[i])：前一天也是dp1状态，或者前一天是dp0状态，今天买入一笔变成dp1状态
dp2 = max(dp2, dp1 + prices[i])：前一天也是dp2状态，或者前一天是dp1状态，今天卖出一笔变成dp2状态
最后答案应是 dp0 和 dp2 的最大值

```python3
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        dp0 = 0
        dp1 = -prices[0]
        dp2 = float("-inf")
        for i in range(1, len(prices)):
            dp0 = 0
            dp1 = max(dp1, dp0 - prices[i])
            dp2 = max(dp2, dp1 + prices[i])
        return max(dp0, dp2)
```

### 8. 整数拆分

[Leetcode-561-medium](https://leetcode-cn.com/problems/integer-break/description/)

**题目描述**：给定一个正整数 n，将其拆分为至少两个正整数的和，并使这些整数的乘积最大化。 返回可以获得的最大乘积。

**定义dp**：dp[i] 为数字 i 可以拆分得到的最大乘积。

**状态转移**：某一个数字的拆分最大乘积可能有两种情况：1. 由 i-j 的拆分最大值乘以 i-j 得到，相当于把 i 拆成 i-j 和一堆组成 j 的数字。2. 由 i-j 乘以 j 得到，相当于把 i 拆成 i-j 和 j。转移方程为dp[i] = max(dp[i], (i - j)*j, (i-j)*dp[i-j])

```python3
class Solution:
    def integerBreak(self, n: int) -> int:
        dp = [0]*(n+1)
        for i in range(2, n+1):
            for j in range(i):
                dp[i] = max(dp[j]*(i-j), j*(i-j), dp[i])
        #print(dp)
        return dp[-1]
```

### 9. 最长递增子序列

[Leetcode-300-medium](https://leetcode-cn.com/problems/longest-increasing-subsequence/description/)

**tips**：此题比[最长连续递增序列](#10-最长连续递增序列)要难

**题目描述**：给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。

**定义dp**：dp[i] 为第 i 个数(该子序列包括 i )之前最长递增子序列。

**状态转移**：如果第 i 个数比第 j 个数大，dp[i] = dp[j] + 1。如果不满足，dp[i] = 1。转移方程：if (nums[i] > nums[j]) dp[i] = max(dp[i], dp[j] + 1)。答案为 max(dp)。

```python3
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        dp = [1]*len(nums)
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[j]+1, dp[i])
        #print(dp)
        return max(dp)
```

### 10. 最长连续递增序列

[Leetcode-674-Easy](https://leetcode-cn.com/problems/longest-continuous-increasing-subsequence/description/)

**题目描述**：给定一个未经排序的整数数组，找到最长且连续递增的子序列，并返回该序列的长度。

**定义dp**：dp[i] 为第 i 个数(该子序列包括 i )之前最长连续递增子序列。

**状态转移**：如果第 i 个数比第 i-1 个数大，dp[i] = dp[i-1] + 1。如果不满足，dp[i] = 1。转移方程：if (nums[i] > nums[i-1]) dp[i] = dp[i-1] + 1。
答案为 max(dp)。

O(N) 时间和空间复杂度：

```python3
class Solution:
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        dp = [1]*len(nums)
        for i in range(1, len(nums)):
            if nums[i] > nums[i-1]:
                dp[i] = dp[i-1]+1
        return max(dp)
```

O(N) 时间复杂度和 O(1) 空间复杂度：

```python3
class Solution:
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        if nums == []:
            return 0
        cur = 1
        res = 1
        for i in range(1, len(nums)):
            if nums[i] > nums[i-1]:
                cur =  cur + 1
            else:
                cur = 1
            res = max(res, cur)
        return res
```

### 11. 最长重复子数组

[Leetcode-718-Medium](https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/description/)

**题目描述**：给两个整数数组 A 和 B ，返回两个数组中公共的、长度最长的子数组的长度。

**定义dp**：dp[i][j] 为数组 A 前 i 个数(包含第 i 个)和数组 B 前 j 个数(包含第 j 个)所构成最长子数组长度。

**状态转移**：dp[i][j] 可能的取值要根据数组 A, B 在 i, j 处的值分类讨论。具体的转移方程如下：if (A[i] == B[j]) dp[i][j] = dp[i-1][j-1]+1 ~~else dp[i][j] = max(dp[i-1][j], dp[i][j-1])~~(这里要求子数组连续)

**初始条件**：先初始 dp[0][j] 和 dp[i][0]。有元素相等则为 1 否则为 0。

```python3
class Solution:
    def findLength(self, A: List[int], B: List[int]) -> int:
        dp = [[0 for i in range(len(B))] for j in range(len(A))]
        res = 0
        # 初始化---------------------------
        for i in range(len(A)):
            if A[i] == B[0]:
                dp[i][0] = 1
        for j in range(len(B)):
            if B[j] == A[0]:
                dp[0][j] = 1
        #---------------------------------
        for i in range(1, len(A)):
            for j in range(1, len(B)):
                if A[i] == B[j]:
                    dp[i][j] = dp[i-1][j-1] + 1
                res = max(dp[i][j], res)
        return res
```

### 12. 最长公共子序列

[Leetcode-1143-Medium](https://leetcode-cn.com/problems/longest-common-subsequence/description/)

**tips**：穿上衣服立马变成[不相交的线](#13-不相交的线)

**题目描述**：给定两个字符串 text1 和 text2，返回这两个字符串的最长公共子序列的长度。

**定义dp**：dp[i][j] 为数组 text1 前 i 个数(包含第 i 个)和数组 text2 前 j 个数(包含第 j 个)所包含的最长公共子序列的长度。

**状态转移**：dp[i][j] 可能的取值要根据 text1[i] 和 text2[j] 的取值分类讨论。具体的转移方程为 if text1[i] == text2[j]: dp[i][j] = dp[i-1][j-1] + 1 else dp[i][j] = max(dp[i-1][j], dp[i][j-1])

**初始条件**：dp[0][j] 和 dp[i][0] 都取 0 即可。

```python3
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        dp  = [[0 for i in range(len(text2)+1)] for j in range(len(text1)+1)]
        for i in range(1, len(text1)+1):
            for j in range(1, len(text2)+1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]
```

### 13. 不相交的线

[Leetcode-1035-Medium](https://leetcode-cn.com/problems/uncrossed-lines/)

**tips**：脱掉衣服立马变成[最长公共子序列](#12-最长公共子序列)

**题目描述**：我们在两条独立的水平线上按给定的顺序写下 A 和 B 中的整数。
现在，我们可以绘制一些连接两个数字 A[i] 和 B[j] 的直线，只要 A[i] == B[j]，且我们绘制的直线不与任何其他连线（非水平线）相交。
以这种方法绘制线条，并返回我们可以绘制的最大连线数。

**定义dp**：dp[i][j] 为数组 A 前 i 个数(包含第 i 个)和数组 B 前 j 个数(包含第 j 个)所包含的满足题意得最大连线数。

**状态转移**：dp[i][j] 可能的取值要根据 A[i] 和 B[j] 的取值分类讨论。具体的转移方程为 if A[i] == B[j]: dp[i][j] = dp[i-1][j-1] + 1 else dp[i][j] = max(dp[i-1][j], dp[i][j-1])

```python3
class Solution:
    def maxUncrossedLines(self, A: List[int], B: List[int]) -> int:
        dp  = [[0 for i in range(len(A)+1)] for j in range(len(B)+1)]
        for i in range(1, len(B)+1):
            for j in range(1, len(A)+1):
                if B[i-1] == A[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]
```

### 14. 判断子序列

[Leetcode-392-Easy](https://leetcode-cn.com/problems/is-subsequence/description/)

**tips**：题目原型[最长公共子序列](#12-最长公共子序列)

**题目描述**：给定字符串 s 和 t ，判断 s 是否为 t 的子序列。

**定义dp**：dp[i][j] 为数组 s 前 i 个数(包含第 i 个)和数组 t 前 j 个数(包含第 j 个)所包含最长公共子序列的长度。

**状态转移**：dp[i][j] 可能的取值要根据 s[i] 和 t[j] 的取值分类讨论。具体的转移方程为 if s[i] == t[j]: dp[i][j] = dp[i-1][j-1] + 1 else dp[i][j] = max(dp[i-1][j], dp[i][j-1])

```python3
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        dp = [[0 for i in range(len(s)+1)] for j in range(len(t)+1)]
        for i in range(1, len(t)+1):
            for j in range(1, len(s)+1):
                if t[i-1] == s[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1] == len(s)
```

### 15. 不同的子序列

[Leetcode-115-Hard](https://leetcode-cn.com/problems/distinct-subsequences/description/)

### 16. 两个字符串的删除操作

[Leetcode-583-Medium](https://leetcode-cn.com/problems/delete-operation-for-two-strings/description/)

