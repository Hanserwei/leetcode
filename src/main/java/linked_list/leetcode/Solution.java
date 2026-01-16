package linked_list.leetcode;

import java.util.*;

class Solution {
    /**
     * 找到两个有序数组中和最小的 k 个数对
     * <p>
     * 算法思路：
     * 1. 使用最小堆来维护当前可能的最小数对
     * 2. 初始时将nums1中每个元素与nums2[0]组成的数对加入堆中
     * 3. 每次从堆中取出和最小的数对，然后将该数对中nums1元素对应的nums2中下一个元素组成的新数对加入堆中
     * 4. 重复k次或堆为空时结束
     * <p>
     * 时间复杂度：O(k * log(min(n, k)))，其中n是nums1的长度
     * 空间复杂度：O(min(n, k))
     *
     * @param nums1 第一个有序数组
     * @param nums2 第二个有序数组
     * @param k     需要返回的数对数量
     * @return 和最小的 k 个数对
     */
    public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        // 创建最小堆，比较器根据数对和进行排序
        // a[0]和b[0]是nums1中的索引，a[1]和b[1]是nums2中的索引
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> (nums1[a[0]] + nums2[a[1]])));

        // 优化策略：只初始化前min(nums1.length, k)个元素
        // 因为我们最多只需要k个结果，所以不需要将所有nums1元素都加入堆中
        for (int i = 0; i < Math.min(nums1.length, k); i++) {
            // 将{nums1索引, nums2索引}加入堆中
            // 初始时nums2索引都是0，表示与nums2的第一个元素配对
            pq.offer(new int[]{i, 0});
        }

        // 存储最终结果的列表
        List<List<Integer>> ans = new ArrayList<>();

        // 循环直到取够 k 个数对或堆为空
        while (!pq.isEmpty() && k-- > 0) {
            // 取出当前和最小的数对索引
            int[] cur = pq.poll();
            int i = cur[0]; // nums1中的索引
            int j = cur[1]; // nums2中的索引

            // 构造数对并添加到结果中
            List<Integer> pair = new ArrayList<>();
            pair.add(nums1[i]); // 添加nums1中的元素
            pair.add(nums2[j]); // 添加nums2中的元素
            ans.add(pair);

            // 如果当前nums2元素不是最后一个，
            // 则将同一nums1元素与nums2的下一个元素组成的新数对加入堆中
            if (j + 1 < nums2.length) {
                pq.offer(new int[]{i, j + 1});
            }
        }

        return ans;
    }

    /**
     * 将两个逆序存储的非负整数链表相加，返回结果链表
     * <p>
     * 算法思路：
     * 1. 使用虚拟头节点简化链表操作
     * 2. 同时遍历两个链表，逐位相加
     * 3. 处理进位：当前位的值 = (val1 + val2 + 进位) % 10
     * 4. 进位值 = (val1 + val2 + 进位) / 10
     * 5. 当两个链表都遍历完且无进位时结束
     * <p>
     * 时间复杂度：O(max(m,n))，其中m和n分别是两个链表的长度
     * 空间复杂度：O(max(m,n))，用于存储结果链表
     *
     * @param l1 第一个数字链表（逆序存储）
     * @param l2 第二个数字链表（逆序存储）
     * @return 两数之和的链表（逆序存储）
     */
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        // 初始化两个指针分别指向两个输入链表
        ListNode p1 = l1, p2 = l2;

        // 创建虚拟头节点，简化链表操作（避免处理头节点的特殊情况）
        ListNode dummyHead = new ListNode(-1);
        // 当前指针用于构建结果链表
        ListNode current = dummyHead;

        // 进位值，初始为0
        int carry = 0;

        // 循环条件：任一链表未遍历完 或 存在进位
        // 注意：即使两个链表都遍历完了，但如果最后一位相加产生进位，仍需继续处理
        while (p1 != null || p2 != null || carry > 0) {
            // 当前位的初始值设为进位值
            int sum = carry;

            // 如果l1链表还有节点，将其值加到sum中
            if (p1 != null) {
                sum += p1.val;
                p1 = p1.next;  // 移动到下一个节点
            }

            // 如果l2链表还有节点，将其值加到sum中
            if (p2 != null) {
                sum += p2.val;
                p2 = p2.next;  // 移动到下一个节点
            }

            // 计算新的进位值（十位数）
            carry = sum / 10;
            // 计算当前位的值（个位数）
            sum %= 10;

            // 创建新节点存储当前位的结果
            current.next = new ListNode(sum);
            // 移动当前指针到新节点
            current = current.next;
        }

        // 返回结果链表（跳过虚拟头节点）
        return dummyHead.next;
    }

    /**
     * 使用栈结构实现两个正序存储的非负整数链表相加
     * <p>
     * 算法思路：
     * 1. 利用栈的后进先出特性，将两个链表的节点值分别压入两个栈中
     * 2. 从栈顶开始弹出元素进行相加运算（相当于从低位到高位计算）
     * 3. 使用头插法构建结果链表，确保结果也是正序存储
     * 4. 处理进位逻辑与addTwoNumbers方法相同
     * <p>
     * 适用场景：当链表是正序存储时（如题目445），此方法比递归更直观
     * 时间复杂度：O(max(m,n))，其中m和n分别是两个链表的长度
     * 空间复杂度：O(m+n)，用于存储两个栈
     *
     * @param l1 第一个数字链表（正序存储）
     * @param l2 第二个数字链表（正序存储）
     * @return 两数之和的链表（正序存储）
     */
    public ListNode addTwoNumbers2(ListNode l1, ListNode l2) {
        // 创建两个栈分别存储两个链表的节点值
        // 栈的特点是后进先出，这样弹出时就是从低位到高位的顺序
        Stack<Integer> stack1 = new Stack<>();
        Stack<Integer> stack2 = new Stack<>();

        // 遍历两个链表，将节点值压入对应栈中
        // 这样栈顶元素就是各自链表的最低位数字
        while (l1 != null || l2 != null) {
            // 将l1的当前节点值压入stack1
            if (l1 != null) {
                stack1.push(l1.val);
                l1 = l1.next;  // 移动到下一个节点
            }

            // 将l2的当前节点值压入stack2
            if (l2 != null) {
                stack2.push(l2.val);
                l2 = l2.next;  // 移动到下一个节点
            }

            // 当两个链表都遍历完成时退出循环
            // 这个判断其实可以省略，因为while条件已经能保证正确退出
            if (l1 == null && l2 == null) {
                break;
            }
        }

        // 创建虚拟头节点，便于构建结果链表
        ListNode dummyHead = new ListNode(-1);
        // 进位值，初始为0
        int carry = 0;

        // 从栈中弹出元素进行相加运算
        // 当任一栈不为空或存在进位时继续循环
        while (!stack1.isEmpty() || !stack2.isEmpty() || carry > 0) {
            // 当前位的初始值设为进位值
            int sum = carry;

            // 从stack1弹出元素（相当于获取l1的当前位）
            if (!stack1.isEmpty()) {
                sum += stack1.pop();
            }

            // 从stack2弹出元素（相当于获取l2的当前位）
            if (!stack2.isEmpty()) {
                sum += stack2.pop();
            }

            // 计算新的进位值（十位数）
            carry = sum / 10;
            // 计算当前位的值（个位数）
            sum %= 10;

            // 使用头插法创建新节点并插入到结果链表头部
            // 这样可以保证结果链表是正序存储的
            ListNode node = new ListNode(sum);
            node.next = dummyHead.next;  // 新节点指向原来的第一个节点
            dummyHead.next = node;       // 新节点成为新的第一个节点
        }

        // 返回结果链表（跳过虚拟头节点）
        return dummyHead.next;
    }

    /**
     * 计算矩阵块和：对于矩阵中每个位置(i,j)，计算以该位置为中心、半径为k的矩形区域内所有元素的和
     * <p>
     * 算法思路：
     * 1. 预处理：使用NumMatrix类构建二维前缀和数组，实现O(1)时间复杂度的矩形区域求和
     * 2. 遍历：对原矩阵中每个位置(i,j)，确定其对应的矩形区域边界
     * 3. 查询：利用前缀和数组快速计算矩形区域内元素和
     * <p>
     * 时间复杂度：O(m*n)，其中m和n分别是矩阵的行数和列数
     * 空间复杂度：O(m*n)，用于存储结果矩阵和前缀和数组
     *
     * @param mat 输入的二维矩阵
     * @param k   矩形区域的半径（中心位置到边界的距离）
     * @return 结果矩阵，其中ans[i][j]表示原矩阵中以(i,j)为中心、半径为k的矩形区域内所有元素的和
     */
    public int[][] matrixBlockSum(int[][] mat, int k) {
        int m = mat.length, n = mat[0].length; // 获取矩阵的行数和列数
        // 创建NumMatrix对象，预处理原矩阵构建二维前缀和数组
        // 这样可以在O(1)时间内查询任意矩形区域的和
        NumMatrix numMatrix = new NumMatrix(mat);
        int[][] ans = new int[m][n]; // 初始化结果矩阵

        // 遍历原矩阵的每个位置(i,j)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                // 计算以(i,j)为中心、半径为k的矩形区域的边界坐标
                // x1, y1: 矩形区域的左上角坐标（考虑边界情况，不能小于0）
                int x1 = Math.max(i - k, 0);
                int y1 = Math.max(j - k, 0);
                // x2, y2: 矩形区域的右下角坐标（考虑边界情况，不能超过矩阵边界）
                int x2 = Math.min(i + k, m - 1);
                int y2 = Math.min(j + k, n - 1);

                // 利用NumMatrix的sumRegion方法快速计算矩形区域[x1,y1]到[x2,y2]内所有元素的和
                // 并将结果存储到ans[i][j]中
                ans[i][j] = numMatrix.sumRegion(x1, y1, x2, y2);
            }
        }
        return ans; // 返回结果矩阵
    }

    /**
     * 寻找数组的中心下标（左侧元素和等于右侧元素和的下标）
     * <p>
     * 算法思路：
     * 1. 使用前缀和数组NumArray预处理原数组，实现O(1)时间复杂度的区间求和
     * 2. 遍历数组的每个位置i，检查左侧[0, i-1]的和是否等于右侧[i+1, n-1]的和
     * 3. 找到满足条件的下标立即返回，遍历完未找到则返回-1
     * <p>
     * 时间复杂度：O(n)，其中n是数组长度
     * 空间复杂度：O(n)，用于存储前缀和数组
     *
     * @param nums 输入的整数数组
     * @return 中心下标，如果不存在则返回-1
     */
    public int pivotIndex(int[] nums) {
        // 创建前缀和数组对象，用于快速计算任意区间的和
        // NumArray内部维护一个前缀和数组preSum，其中preSum[i]表示nums[0]到nums[i]的累加和
        NumArray numArray = new NumArray(nums);
        int n = nums.length; // 获取数组长度

        // 遍历数组的每个位置，检查是否为中心下标
        for (int i = 0; i < n; i++) {
            // 计算左侧[0, i-1]的元素和
            // sumRange(0, i)表示从索引0到索引i-1的区间和
            int leftSum = numArray.sumRange(0, i - 1);

            // 计算右侧[i+1, n-1]的元素和
            // sumRange(i + 1, n - 1)表示从索引i+1到索引n-1的区间和
            int rightSum = numArray.sumRange(i + 1, n - 1);

            // 如果左侧和等于右侧和，则当前下标i为中心下标
            if (leftSum == rightSum) {
                return i; // 找到中心下标，立即返回
            }
        }

        // 遍历完所有位置都未找到中心下标，返回-1
        return -1;
    }

    /**
     * 计算数组中除当前元素外其余元素的乘积
     * <p>
     * 算法思路：
     * 1. 使用两个辅助数组分别存储前缀乘积和后缀乘积
     * 2. 前缀数组prefix[i]表示nums[0]到nums[i]的乘积
     * 3. 后缀数组suffix[i]表示nums[i]到nums[n-1]的乘积
     * 4. 对于位置i的结果ans[i] = prefix[i-1] * suffix[i+1]
     * 5. 即：左侧所有元素乘积 × 右侧所有元素乘积
     * <p>
     * 时间复杂度：O(n)，其中n是数组长度（三次遍历）
     * 空间复杂度：O(n)，用于存储前缀数组、后缀数组和结果数组
     *
     * @param nums 输入的整数数组
     * @return 结果数组，其中ans[i]表示除nums[i]外所有元素的乘积
     */
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length; // 获取数组长度

        // 构建前缀乘积数组：prefix[i]表示从nums[0]到nums[i]所有元素的乘积
        int[] prefix = new int[n];
        prefix[0] = nums[0]; // 第一个元素的前缀乘积就是它本身
        // 从第二个元素开始，每个位置的前缀乘积 = 前一个位置的前缀乘积 × 当前元素
        for (int i = 1; i < n; i++) {
            prefix[i] = prefix[i - 1] * nums[i];
        }

        // 构建后缀乘积数组：suffix[i]表示从nums[i]到nums[n-1]所有元素的乘积
        int[] suffix = new int[n];
        suffix[n - 1] = nums[n - 1]; // 最后一个元素的后缀乘积就是它本身
        // 从倒数第二个元素开始向前，每个位置的后缀乘积 = 后一个位置的后缀乘积 × 当前元素
        for (int i = n - 2; i >= 0; i--) {
            suffix[i] = suffix[i + 1] * nums[i];
        }

        // 构建结果数组
        int[] ans = new int[n];

        // 处理边界情况：第一个元素的结果 = 除它之外的所有元素乘积 = suffix[1]
        // 因为suffix[1]表示从nums[1]到nums[n-1]的乘积（即右侧所有元素的乘积）
        ans[0] = suffix[1];

        // 处理边界情况：最后一个元素的结果 = 除它之外的所有元素乘积 = prefix[n-2]
        // 因为prefix[n-2]表示从nums[0]到nums[n-2]的乘积（即左侧所有元素的乘积）
        ans[n - 1] = prefix[n - 2];

        // 处理中间元素：对于位置i，结果 = 左侧所有元素乘积 × 右侧所有元素乘积
        // 左侧所有元素乘积 = prefix[i-1]（前缀数组中位置i-1的值）
        // 右侧所有元素乘积 = suffix[i+1]（后缀数组中位置i+1的值）
        for (int i = 1; i < n - 1; i++) {
            ans[i] = prefix[i - 1] * suffix[i + 1];
        }

        return ans; // 返回结果数组
    }

    /**
     * 寻找最长连续子数组，使得其中0和1的数量相等
     * <p>
     * 算法思路：
     * 1. 转换：将数组中的0转换为-1，这样问题就转化为寻找和为0的最长子数组
     * 2. 前缀和：使用前缀和数组记录从开始到当前位置的累计和
     * 3. 哈希表：利用哈希表记录每个前缀和第一次出现的位置
     * 4. 匹配：当相同的前缀和再次出现时，说明这两个位置之间的子数组和为0
     * <p>
     * 核心思想：如果preSum[i] == preSum[j]（i < j），则区间[i+1, j]的和为0，
     * 即该区间内0和1的数量相等。
     * <p>
     * 时间复杂度：O(n)，其中n是数组长度
     * 空间复杂度：O(n)，用于存储前缀和数组和哈希表
     *
     * @param nums 输入的二进制数组（只包含0和1）
     * @return 满足条件的最长子数组的长度
     */
    public int findMaxLength(int[] nums) {
        int n = nums.length; // 获取数组长度

        // 构建前缀和数组：长度为n+1便于处理边界情况
        // preSum[i]表示从nums[0]到nums[i-1]的前缀和（经过0转-1的转换）
        int[] preSum = new int[n + 1];
        preSum[0] = 0; // 前缀和数组的初始值为0（空数组的和）

        // 构建前缀和数组：将0视为-1，1保持不变
        // 这样如果某个子数组中0和1数量相等，那么其和就为0
        for (int i = 1; i <= n; i++) {
            // 如果当前元素是0，则在前缀和基础上减1；如果是1，则加1
            preSum[i] = preSum[i - 1] + (nums[i - 1] == 0 ? -1 : nums[i - 1]);
        }

        // 创建哈希表，用于存储前缀和值到其首次出现索引的映射
        // key: 前缀和值，value: 该前缀和首次出现的索引位置
        HashMap<Integer, Integer> valToIndex = new HashMap<>();

        int res = 0; // 记录最长符合条件子数组的长度

        // 遍历前缀和数组，寻找相同前缀和值的最远距离
        for (int i = 0; i < preSum.length; i++) {
            // 如果当前前缀和值之前没有出现过
            if (!valToIndex.containsKey(preSum[i])) {
                // 将该前缀和值与其索引建立映射关系
                // 只记录第一次出现的位置，这样能保证后续计算的距离最大
                valToIndex.put(preSum[i], i);
            } else {
                // 如果当前前缀和值之前出现过
                // 说明从上次出现的位置到当前位置之间的子数组和为0
                // 即该子数组中0和1的数量相等
                // 计算当前子数组长度并更新最大长度
                res = Math.max(res, i - valToIndex.get(preSum[i]));
            }
        }

        return res; // 返回最长符合条件子数组的长度
    }

    /**
     * 检查数组中是否存在长度至少为2的子数组，其元素和为k的倍数
     * <p>
     * 算法思路：
     * 1. 使用前缀和数组计算累积和
     * 2. 利用同余定理：如果两个前缀和对k取模相等，则它们之间的子数组和一定是k的倍数
     * 3. 使用哈希表记录每个余数首次出现的位置，当再次遇到相同余数时，检查距离是否>=2
     * <p>
     * 核心原理：如果preSum[i] % k == preSum[j] % k (i < j)，则(preSum[j] - preSum[i]) % k == 0，
     * 即区间[i+1, j]的和是k的倍数。
     *
     * @param nums 输入的整数数组
     * @param k    给定的整数
     * @return 如果存在满足条件的子数组返回true，否则返回false
     */
    public boolean checkSubarraySum(int[] nums, int k) {
        int n = nums.length; // 获取数组长度

        // 构建前缀和数组：长度为n+1便于处理边界情况
        // preSum[i]表示从nums[0]到nums[i-1]的前缀和
        int[] preSum = new int[n + 1];
        preSum[0] = 0; // 前缀和数组的初始值为0（空数组的和）

        // 构建前缀和数组：计算从数组开始到每个位置的累积和
        for (int i = 1; i <= n; i++) {
            preSum[i] = preSum[i - 1] + nums[i - 1];
        }

        // 创建哈希表，用于存储前缀和对k取模的结果到其首次出现索引的映射
        // key: 前缀和 % k 的余数，value: 该余数首次出现的索引位置
        HashMap<Integer, Integer> valToIndex = new HashMap<>();

        // 第一次遍历：记录每个余数首次出现的位置
        for (int i = 0; i < preSum.length; i++) {
            int val = preSum[i] % k; // 计算当前前缀和对 k 取模的余数
            if (!valToIndex.containsKey(val)) {
                // 只记录余数首次出现的位置，这样能保证后续计算的距离最大
                valToIndex.put(val, i);
            }
        }

        // 第二次遍历：查找相同余数的最远距离，检查是否存在长度>=2的子数组
        for (int i = 0; i < preSum.length; i++) {
            int need = preSum[i] % k; // 当前前缀和对 k 取模的余数

            // 如果该余数之前出现过，说明找到了两个前缀和对k取模相等的情况
            if (valToIndex.containsKey(need)) {
                // 计算两个相同余数之间的距离（即子数组的长度）
                // i - valToIndex.get(need) 表示从首次出现该余数的位置到当前位置的距离
                if (i - valToIndex.get(need) >= 2) {
                    // 如果距离大于等于2，说明找到了长度至少为2的子数组，其和为k的倍数
                    return true;
                }
            }
        }

        // 遍历完所有位置都没有找到满足条件的子数组，返回false
        return false;
    }

    /**
     * 计算数组中和为 k 的连续子数组的个数
     * <p>
     * 算法思路：
     * 1. 使用前缀和 + 哈希表优化：遍历数组过程中维护前缀和，并用哈希表记录每个前缀和出现的次数
     * 2. 对于当前前缀和preSum，如果存在之前的前缀和等于preSum - k，则说明存在子数组和为k
     * 3. 核心原理：如果preSum[j] - preSum[i] = k，则区间[i+1, j]的和为k
     *
     * @param nums 输入的整数数组
     * @param k    目标和
     * @return 和为 k 的连续子数组的个数
     */
    public int subarraySum(int[] nums, int k) {
        // 创建哈希表存储前缀和及其出现次数的映射
        // key: 前缀和值，value: 该前缀和出现的次数
        Map<Integer, Integer> countMap = new HashMap<>();

        // 初始化：前缀和为0出现1次（对应空数组的情况）
        // 这是为了处理从数组开始到某位置的子数组和正好等于k的情况
        countMap.put(0, 1);

        // 记录满足条件的子数组个数
        int res = 0;

        // 当前前缀和，表示从数组开始到当前位置的累积和
        int preSum = 0;

        // 遍历数组中的每个元素
        for (int num : nums) {
            // 更新前缀和：加上当前元素
            preSum += num;

            // 计算需要寻找的目标前缀和
            // 如果存在前缀和为(preSum - k)的情况，则说明存在子数组和为k
            // 因为：当前前缀和 - 目标前缀和 = k → 目标前缀和 = 当前前缀和 - k
            int need = preSum - k;

            // 如果哈希表中存在目标前缀和，说明找到了若干个和为k的子数组
            // 将对应的出现次数加到结果中
            if (countMap.containsKey(need)) {
                res += countMap.get(need);
            }

            // 将当前前缀和加入哈希表，更新其出现次数
            // getOrDefault方法获取当前前缀和的出现次数，如果不存在则默认为0
            countMap.put(preSum, countMap.getOrDefault(preSum, 0) + 1);
        }

        return res; // 返回和为 k 的连续子数组的个数
    }

    /**
     * 寻找最长的"表现良好"时间段
     * <p>
     * "表现良好"的定义：工作小时数严格大于8小时的天数 > 工作小时数小于等于8小时的天数
     * 算法思路：
     * 1. 转换问题：将hours[i] > 8的天数标记为+1，hours[i] <= 8的天数标记为-1
     * 2. 问题转化为：寻找和大于0的最长子数组（因为+1多于-1时总和才大于0）
     * 3. 使用前缀和 + 哈希表优化：
     * - 前缀和preSum[i]表示从开始到第i天的累积分数差值
     * - 如果preSum[i] > 0，说明从开始到第i天整体表现良好
     * - 如果preSum[i] <= 0，需要找到最早的j，使得preSum[i] - preSum[j] > 0，即preSum[j] < preSum[i]
     * - 为了最大化长度，我们只记录每个前缀和值第一次出现的位置
     * <p>
     * 核心思想：如果preSum[i] - preSum[j] > 0，则区间[j+1, i]是表现良好的时间段
     * <p>
     * 时间复杂度：O(n)，其中n是数组长度
     * 空间复杂度：O(n)，用于存储前缀和数组和哈希表
     *
     * @param hours 工作小时数组，hours[i]表示第i天的工作小时数
     * @return 最长"表现良好"时间段的长度
     */
    public int longestWPI(int[] hours) {
        int n = hours.length; // 获取工作天数

        // 构建前缀和数组：长度为n+1便于处理边界情况
        // preSum[i]表示从第1天到第i天的累积分数差值（>8小时记为+1，≤8小时记为-1）
        int[] preSum = new int[n + 1];
        preSum[0] = 0; // 前缀和数组的初始值为0（空时间段的差值为0）

        // 创建哈希表，用于存储前缀和值到其首次出现索引的映射
        // key: 前缀和值，value: 该前缀和首次出现的天数索引
        // 只记录首次出现的位置，这样能保证后续计算的时间段长度最大
        Map<Integer, Integer> valToIndex = new HashMap<>();

        int res = 0; // 记录最长"表现良好"时间段的长度

        // 遍历每一天，构建前缀和并寻找最长的正数区间
        for (int i = 1; i < n + 1; i++) {
            // 计算当前前缀和：如果当天工作时间>8小时则+1，否则-1
            // 这样如果某段时间内+1比-1多，前缀和就为正，表示这段时间表现良好
            preSum[i] = preSum[i - 1] + (hours[i - 1] > 8 ? 1 : -1);

            // 如果当前前缀和值之前没有出现过，记录其首次出现的位置
            // 只记录第一次出现的位置，保证后续计算的距离最大
            if (!valToIndex.containsKey(preSum[i])) {
                valToIndex.put(preSum[i], i);
            }

            // 情况1：如果当前前缀和大于0，说明从第1天到第i天整体表现良好
            if (preSum[i] > 0) {
                // 更新最长时间段长度为i（从第1天到第i天共i天）
                res = Math.max(res, i);
            } else {
                // 情况2：如果当前前缀和不大于0，尝试找到一个较早的位置j
                // 使得preSum[i] - preSum[j] > 0，即preSum[j] < preSum[i]
                // 由于我们只关心preSum[j] < preSum[i]的情况，而preSum[i]是整数
                // 所以最接近的可能是preSum[j] = preSum[i] - 1
                if (valToIndex.containsKey(preSum[i] - 1)) {
                    // 找到最早出现preSum[i] - 1的位置j
                    int j = valToIndex.get(preSum[i] - 1);
                    // 区间[j+1, i]的前缀和差值为preSum[i] - preSum[j] = preSum[i] - (preSum[i] - 1) = 1 > 0
                    // 所以这段区间是表现良好的，长度为i - j
                    res = Math.max(res, i - j);
                }
            }
        }

        return res; // 返回最长"表现良好"时间段的长度
    }

    /**
     * 计算航班预定座位数
     * <p>
     * 算法思路：
     * 1. 使用差分数组优化区间更新操作
     * 2. 每个预订记录[begin, end, seats]表示在区间[begin, end]上增加seats个座位
     * 3. 利用差分数组的特性，将区间更新的时间复杂度从O(n)降到O(1)
     * 4. 最后通过前缀和还原得到最终结果
     * <p>
     * 时间复杂度：O(m + n)，其中m是预订记录数，n是航班数
     * 空间复杂度：O(n)，用于存储差分数组
     *
     * @param bookings 预订记录数组，每个元素为[begin, end, seats]，表示在航班[begin, end]上预订seats个座位
     * @param n        航班总数（航班编号从1到n）
     * @return 每个航班的座位预订总数
     */
    public int[] corpFlightBookings(int[][] bookings, int n) {
        // 创建长度为n的数组，用于存储每个航班的座位变化情况
        // 注意：这里应该创建长度为n的数组而不是固定长度5，因为航班数量是动态的
        int[] res = new int[n];

        // 创建差分数组对象，用于高效处理区间更新操作
        Difference df = new Difference(res);

        // 遍历所有预订记录
        for (int[] booking : bookings) {
            // 获取预订的起始航班（转换为0-based索引）
            int i = booking[0] - 1;
            // 获取预订的结束航班（转换为0-based索引）
            int j = booking[1] - 1;
            // 获取预订的座位数
            int k = booking[2];

            // 在差分数组的区间[i, j]上增加k个座位
            // 这样可以在O(1)时间内完成区间更新
            df.increment(i, j, k);
        }

        // 通过前缀和还原原始数组，得到每个航班的最终座位数
        return df.result();
    }

    /**
     * 判断是否能够完成所有行程的拼车需求
     * <p>
     * 算法思路：
     * 1. 使用差分数组来高效处理区间更新操作
     * 2. 每个行程trip[0]表示乘客数量，trip[1]表示上车地点，trip[2]表示下车地点
     * 3. 在区间[上车地点, 下车地点-1]上增加乘客数量（下车地点不包含，因为乘客已在该点下车）
     * 4. 通过前缀和还原数组，检查每个地点的人数是否超过容量限制
     * <p>
     * 时间复杂度：O(n + m)，其中n是行程数量，m是地点范围（这里是1001）
     * 空间复杂度：O(m)，用于存储差分数组
     *
     * @param trips    行程数组，每个元素为[乘客数量, 上车地点, 下车地点]
     * @param capacity 车辆容量
     * @return 如果能够在不超载的情况下完成所有行程返回true，否则返回false
     */
    public boolean carPooling(int[][] trips, int capacity) {
        // 创建差分数组，处理地点范围0-1000（题目限制最大为1000）
        Difference difference = new Difference(new int[1001]);

        // 遍历每个行程，更新差分数组
        for (int[] trip : trips) {
            int passengerCount = trip[0];  // 当前行程的乘客数量
            int pickupLocation = trip[1];  // 上车地点
            int dropOffLocation = trip[2] - 1;  // 下车地点（减1是因为到达下车地点时乘客已下车）

            // 在上车地点到下车地点前一站之间增加乘客数量
            // 注意：下车地点不包含在区间内，因为乘客在该地点已经下车
            difference.increment(pickupLocation, dropOffLocation, passengerCount);
        }

        // 通过前缀和还原实际的乘客数量分布
        int[] actualPassengerCounts = difference.result();

        // 检查每个地点的乘客数量是否超过车辆容量
        for (int currentPassengerCount : actualPassengerCounts) {
            if (currentPassengerCount > capacity) {
                // 如果某个地点的乘客数量超过了车辆容量，返回false
                return false;
            }
        }

        // 所有地点的乘客数量都不超过容量，返回true
        return true;
    }

    /**
     * 将矩阵顺时针旋转90度
     * <p>
     * 算法思路：
     * 1. 先进行主对角线翻转（转置矩阵）：matrix[i][j] 与 matrix[j][i] 交换
     * 2. 再对每一行进行水平翻转（反转每行元素）
     * <p>
     * 数学原理：
     * - 主对角线翻转：将矩阵按左上到右下的对角线镜像翻转
     * - 水平翻转：将每行的元素顺序完全颠倒
     * - 两步操作的组合效果等价于整个矩阵顺时针旋转90度
     * <p>
     * 示例：
     * 原矩阵：
     * 1 2 3
     * 4 5 6
     * 7 8 9
     * <p>
     * 步骤1 - 主对角线翻转后：
     * 1 4 7
     * 2 5 8
     * 3 6 9
     * <p>
     * 步骤2 - 每行水平翻转后（最终结果）：
     * 7 4 1
     * 8 5 2
     * 9 6 3
     * <p>
     * 时间复杂度：O(n²)，其中n是矩阵的边长
     * 空间复杂度：O(1)，原地操作
     *
     * @param matrix 输入的n×n矩阵
     */
    public void rotate(int[][] matrix) {
        int n = matrix.length; // 获取矩阵的边长

        // 步骤1：沿主对角线翻转矩阵（转置操作）
        // 只需要遍历上三角部分（j从i开始），避免重复交换
        for (int i = 0; i < n; i++) {
            // j从i开始，确保只处理上三角部分，防止同一个位置被交换两次
            for (int j = i; j < n; j++) {
                // 交换matrix[i][j]和matrix[j][i]，实现转置
                int temp = matrix[i][j];      // 临时保存当前位置的值
                matrix[i][j] = matrix[j][i];  // 将对称位置的值赋给当前位置
                matrix[j][i] = temp;          // 将临时保存的值赋给对称位置
            }
        }

        // 步骤2：对每一行进行水平翻转
        // 遍历每一行，调用reverse方法将该行元素顺序颠倒
        for (int[] row : matrix) {
            // 对每一行调用反转方法，实现水平翻转
            reverse(row);
        }
    }

    /**
     * 反转一维数组（双指针法）
     * <p>
     * 算法思路：
     * 使用双指针从数组两端向中间移动，交换对应位置的元素
     * <p>
     * 时间复杂度：O(n)，其中n是数组长度
     * 空间复杂度：O(1)，原地操作
     *
     * @param row 要反转的一维数组
     */
    private void reverse(int[] row) {
        // 初始化双指针：i指向数组开始，j指向数组结束
        int i = 0, j = row.length - 1;

        // 当左指针小于右指针时继续交换
        while (i < j) {
            // 交换位置 i 和位置 j 的元素
            int temp = row[i];    // 临时保存左指针位置的值
            row[i] = row[j];      // 将右指针位置的值赋给左指针位置
            row[j] = temp;        // 将临时保存的值赋给右指针位置

            // 移动指针：左指针右移，右指针左移
            i++;  // 左指针向右移动一位
            j--;  // 右指针向左移动一位
        }
    }

    public void rotate2(int[][] matrix) {
        int n = matrix.length; // 获取矩阵的边长

        for (int i = 0; i < n; i++) {
            // j的范围是0到n-i-1，确保只处理反对角线一侧的元素，避免重复交换
            for (int j = 0; j < n - i; j++) {
                // 交换matrix[i][j]和它的反对角线对称位置matrix[n-1-j][n-1-i]
                int temp = matrix[i][j];
                matrix[i][j] = matrix[n - j - 1][n - i - 1];
                matrix[n - j - 1][n - i - 1] = temp;
            }
        }

        for (int[] row : matrix) {
            // 对每一行调用反转方法，实现水平翻转
            reverse(row);
        }
    }

    /**
     * 删除链表中的重复元素，保留所有不重复的节点
     * <p>
     * 算法思路：
     * 1. 使用两个虚拟头节点分别管理不重复节点和重复节点
     * 2. 遍历原链表，根据当前节点是否重复来决定将其连接到哪个链表
     * 3. 判断重复的条件：
     * - 与下一个节点值相同
     * - 与重复链表的最后一个节点值相同
     * 4. 最终返回不重复节点链表的头节点
     * <p>
     * 时间复杂度：O(n)，其中n是链表长度
     * 空间复杂度：O(1)，只使用了常数个额外节点
     *
     * @param head 链表头节点
     * @return 删除重复元素后的链表头节点
     */
    public ListNode deleteDuplicates(ListNode head) {
        // 创建两个虚拟头节点：
        // dumpyUni：用于收集所有不重复的节点
        // dumpyDup：用于收集所有重复的节点
        // 值设为101是为了避免与链表中可能出现的节点值冲突
        ListNode dumpyUni = new ListNode(101);
        ListNode dumpyDup = new ListNode(101);

        // p1指向重复节点链表的当前末尾
        // p2指向不重复节点链表的当前末尾
        ListNode p1 = dumpyDup;
        ListNode p2 = dumpyUni;

        // cur 用于遍历原链表
        ListNode cur = head;

        // 遍历整个链表
        while (cur != null) {
            // 判断当前节点是否应该放入重复链表：
            // 条件1：当前节点与下一个节点值相同（说明是重复节点）
            // 条件2：当前节点值与重复链表最后一个节点值相同（说明也是重复节点）
            if ((cur.next != null && cur.val == cur.next.val) || cur.val == p1.val) {
                // 将当前节点连接到重复链表末尾
                p1.next = cur;
                p1 = p1.next;  // 更新重复链表的末尾指针
            } else {
                // 当前节点不重复，连接到不重复链表末尾
                p2.next = cur;
                p2 = p2.next;  // 更新不重复链表的末尾指针
            }

            // 移动到下一个节点继续处理
            cur = cur.next;

            // 断开当前节点与原链表的连接，避免形成环
            // 这很重要，确保每个节点只属于一个链表
            p1.next = null;
            p2.next = null;
        }

        // 返回不重复节点链表的头节点（跳过虚拟头节点）
        return dumpyUni.next;
    }


    /**
     * 螺旋顺序遍历二维矩阵
     * <p>
     * 算法思路：
     * 1. 使用四个边界变量控制遍历范围：上边界、下边界、左边界、右边界
     * 2. 按照"右→下→左→上"的顺序依次遍历矩阵的外圈元素
     * 3. 每遍历完一条边，相应的边界就向内收缩一格
     * 4. 重复上述过程直到遍历完所有元素
     * <p>
     * 时间复杂度：O(m*n)，其中m和n分别是矩阵的行数和列数
     * 空间复杂度：O(1)，不考虑结果数组的存储空间
     *
     * @param matrix 输入的二维矩阵
     * @return 螺旋顺序遍历的结果列表
     */
    public List<Integer> spiralOrder(int[][] matrix) {
        // 获取矩阵的行数和列数
        int m = matrix.length;      // 矩阵的行数
        int n = matrix[0].length;   // 矩阵的列数

        // 初始化四个边界变量
        int upper_bound = 0;        // 上边界：初始为第0行
        int lower_bound = m - 1;    // 下边界：初始为最后一行
        int right_bound = n - 1;    // 右边界：初始为最后一列
        int left_bound = 0;         // 左边界：初始为第0列

        // 创建结果列表，用于存储螺旋顺序遍历的元素
        List<Integer> res = new ArrayList<>();

        // 循环直到遍历完所有元素
        // 判断条件：结果列表的大小小于矩阵元素总数
        while (res.size() < m * n) {
            // 第一步：从左到右遍历上边界这一行
            // 前提条件：上边界不能超过下边界（确保还有行需要遍历）
            if (upper_bound <= lower_bound) {
                // 从左边界到右边界遍历上边界行
                for (int j = left_bound; j <= right_bound; j++) {
                    res.add(matrix[upper_bound][j]);
                }
                // 遍历完后，上边界向下移动一行
                upper_bound++;
            }

            // 第二步：从上到下遍历右边界这一列
            // 前提条件：左边界不能超过右边界（确保还有列需要遍历）
            if (left_bound <= right_bound) {
                // 从上边界到下边界遍历右边界列
                // 注意：这里的循环变量应该是i（行索引），而不是原代码中的left_bound
                for (int i = upper_bound; i <= lower_bound; i++) {
                    res.add(matrix[i][right_bound]);
                }
                // 遍历完后，右边界向左移动一列
                right_bound--;
            }

            // 第三步：从右到左遍历下边界这一行
            // 前提条件：上边界不能超过下边界（确保还有行需要遍历）
            if (upper_bound <= lower_bound) {
                // 从右边界到左边界遍历下边界行（逆序）
                for (int j = right_bound; j >= left_bound; j--) {
                    res.add(matrix[lower_bound][j]);
                }
                // 遍历完后，下边界向上移动一行
                lower_bound--;
            }

            // 第四步：从下到上遍历左边界这一列
            // 前提条件：左边界不能超过右边界（确保还有列需要遍历）
            if (left_bound <= right_bound) {
                // 从下边界到上边界遍历左边界列（逆序）
                for (int i = lower_bound; i >= upper_bound; i--) {
                    res.add(matrix[i][left_bound]);
                }
                // 遍历完后，左边界向右移动一列
                left_bound++;
            }
        }

        // 返回螺旋顺序遍历的结果列表
        return res;
    }


    /**
     * 螺旋顺序生成 n x n 矩阵
     * <p>
     * 算法思路：
     * 1. 创建一个 n x n 的空矩阵
     * 2. 使用四个边界变量控制填充范围：上边界、下边界、左边界、右边界
     * 3. 按照"右→下→左→上"的顺序依次填充数字 1, 2, 3, ..., n²
     * 4. 每填充完一条边，相应的边界就向内收缩一格
     * 5. 重复上述过程直到填充完所有位置
     * <p>
     * 与 spiralOrder 方法的对比：
     * - spiralOrder：遍历已有矩阵，按螺旋顺序读取元素
     * - generateMatrix：生成新矩阵，按螺旋顺序填充递增数字
     * <p>
     * 时间复杂度：O(n²)，需要填充 n² 个位置
     * 空间复杂度：O(1)，不考虑结果矩阵的存储空间
     *
     * @param n 矩阵的边长（生成 n x n 的方阵）
     * @return 按螺旋顺序填充数字 1 到 n² 的矩阵
     */
    public int[][] generateMatrix(int n) {
        // 创建 n x n 的空矩阵，所有元素初始值为0
        int[][] matrix = new int[n][n];

        // 初始化四个边界变量
        int upper_bound = 0;        // 上边界：初始为第0行
        int lower_bound = n - 1;    // 下边界：初始为最后一行
        int left_bound = 0;         // 左边界：初始为第0列
        int right_bound = n - 1;    // 右边界：初始为最后一列

        // num用于记录当前要填充的数字，从1开始递增
        int num = 1;

        // 循环直到填充完所有 n² 个位置
        // 判断条件：num <= n * n 表示还有数字未填充
        while (num <= n * n) {
            // 第一步：从左到右填充上边界这一行
            // 前提条件：上边界不能超过下边界（确保还有行需要填充）
            if (upper_bound <= lower_bound) {
                // 从左边界到右边界依次填充上边界行
                for (int j = left_bound; j <= right_bound; j++) {
                    matrix[upper_bound][j] = num++;  // 填充当前数字并递增
                }
                // 填充完后，上边界向下移动一行
                upper_bound++;
            }

            // 第二步：从上到下填充右边界这一列
            // 前提条件：左边界不能超过右边界（确保还有列需要填充）
            if (left_bound <= right_bound) {
                // 从上边界到下边界依次填充右边界列
                for (int i = upper_bound; i <= lower_bound; i++) {
                    matrix[i][right_bound] = num++;  // 填充当前数字并递增
                }
                // 填充完后，右边界向左移动一列
                right_bound--;
            }

            // 第三步：从右到左填充下边界这一行
            // 前提条件：上边界不能超过下边界（确保还有行需要填充）
            if (upper_bound <= lower_bound) {
                // 从右边界到左边界依次填充下边界行（逆序）
                for (int j = right_bound; j >= left_bound; j--) {
                    matrix[lower_bound][j] = num++;  // 填充当前数字并递增
                }
                // 填充完后，下边界向上移动一行
                lower_bound--;
            }

            // 第四步：从下到上填充左边界这一列
            // 前提条件：左边界不能超过右边界（确保还有列需要填充）
            if (left_bound <= right_bound) {
                // 从下边界到上边界依次填充左边界列（逆序）
                for (int i = lower_bound; i >= upper_bound; i--) {
                    matrix[i][left_bound] = num++;  // 填充当前数字并递增
                }
                // 填充完后，左边界向右移动一列
                left_bound++;
            }
        }

        // 返回填充完成的螺旋矩阵
        return matrix;
    }


    /**
     * 反转字符串中的单词顺序
     * <p>
     * 算法思路：
     * 1. 去除首尾空格并处理单词间的多余空格（保留单个空格）
     * 2. 翻转整个字符串（所有字符逆序）
     * 3. 再将每个单词单独翻转（恢复单词内部的正确顺序）
     * <p>
     * 示例：
     * 输入: "  hello world  "
     * 步骤1: "hello world" （去除多余空格）
     * 步骤2: "dlrow olleh" （整体翻转）
     * 步骤3: "world hello" （每个单词单独翻转）
     * <p>
     * 时间复杂度：O(n)，其中n是字符串长度
     * 空间复杂度：O(n)，用于存储StringBuilder
     *
     * @param s 输入的字符串
     * @return 单词顺序反转后的字符串
     */
    public String reverseWords(String s) {
        // 边界检查：如果输入为null，直接返回null
        if (s == null) return null;

        // 步骤1：去掉首尾空格并处理中间多余空格
        // 将字符串规范化：去除首尾空格，单词间只保留一个空格
        StringBuilder sb = trimSpaces(s);

        // 步骤2：翻转整个字符串
        // 这样单词的顺序就反转了，但每个单词内部的字符也被反转了
        reverse(sb, 0, sb.length() - 1);

        // 步骤3：翻转每个单词
        // 将每个单词内部的字符再次反转，恢复单词的正确拼写
        reverseEachWord(sb);

        // 返回最终结果字符串
        return sb.toString();
    }

    /**
     * 去除字符串首尾空格并处理单词间的多余空格
     * <p>
     * 算法步骤：
     * 1. 使用双指针找到字符串的有效范围（去除首尾空格）
     * 2. 遍历有效范围内的字符，去除单词间的多余空格
     * <p>
     * 处理逻辑：
     * - 遇到非空格字符：直接添加
     * - 遇到空格字符：只在前一个字符不是空格时添加（避免连续空格）
     *
     * @param s 原始字符串
     * @return 去除多余空格后的StringBuilder对象
     */
    private StringBuilder trimSpaces(String s) {
        int left = 0, right = s.length() - 1;

        // 去掉字符串开头的所有空格
        // 从左向右移动left指针，直到遇到第一个非空格字符
        while (left <= right && s.charAt(left) == ' ') left++;

        // 去掉字符串末尾的所有空格
        // 从右向左移动right指针，直到遇到最后一个非空格字符
        while (left <= right && s.charAt(right) == ' ') right--;

        // 去掉单词间的多余空格，只保留一个空格
        StringBuilder sb = new StringBuilder();
        while (left <= right) {
            char c = s.charAt(left);

            // 如果当前字符不是空格，直接添加到结果中
            if (c != ' ') {
                sb.append(c);
            }
            // 如果当前字符是空格，且前一个字符不是空格，才添加
            // 这样可以确保单词间只保留一个空格
            else if (sb.charAt(sb.length() - 1) != ' ') {
                sb.append(c); // 只保留一个空格
            }
            // 如果当前是空格且前一个也是空格，不做任何操作（跳过多余空格）

            left++; // 移动到下一个字符
        }
        return sb;
    }

    /**
     * 翻转StringBuilder中指定区间的字符
     * <p>
     * 算法思路：
     * 使用双指针从两端向中间移动，交换对应位置的字符
     * <p>
     * 示例：
     * 输入: "hello", left=0, right=4
     * 过程: h<->o, e<->l, l（中心不动）
     * 输出: "olleh"
     *
     * @param sb    要操作的StringBuilder对象
     * @param left  起始位置（包含）
     * @param right 结束位置（包含）
     */
    private void reverse(StringBuilder sb, int left, int right) {
        // 双指针从两端向中间移动
        while (left < right) {
            // 暂存左指针位置的字符
            char temp = sb.charAt(left);

            // 交换左右指针位置的字符
            sb.setCharAt(left++, sb.charAt(right));   // 将右边字符放到左边，左指针右移
            sb.setCharAt(right--, temp);              // 将左边字符放到右边，右指针左移
        }
        // 当left >= right时，说明所有字符已经交换完成
    }

    /**
     * 翻转StringBuilder中的每个单词
     * <p>
     * 算法步骤：
     * 1. 遍历字符串，找到每个单词的起始和结束位置
     * 2. 对每个单词调用reverse方法进行翻转
     * 3. 更新指针继续查找下一个单词
     * <p>
     * 单词识别：以空格为分隔符，连续的非空格字符组成一个单词
     *
     * @param sb 要操作的StringBuilder对象
     */
    private void reverseEachWord(StringBuilder sb) {
        int n = sb.length();
        int start = 0, end = 0; // start指向单词开始，end用于寻找单词结束

        while (start < n) {
            // 找到当前单词的末尾位置
            // end指针向右移动，直到遇到空格或到达字符串末尾
            while (end < n && sb.charAt(end) != ' ') end++;

            // 此时end指向空格或已超出范围
            // 翻转从start到end-1的单词（end-1是单词的最后一个字符）
            reverse(sb, start, end - 1);

            // 更新指针，准备寻找下一个单词
            start = end + 1;  // 跳过空格，指向下一个单词的开始
            end++;            // end 也同步移动到下一个位置
        }
        // 循环结束时，所有单词都已被翻转
    }

    /**
     * 删除有序数组中的重复项，使每个元素只出现一次，并返回新长度
     * <p>
     * 算法思路：
     * 1. 使用快慢指针（双指针）技巧，在一个有序数组中寻找唯一的元素
     * 2. 慢指针 slow：指向已经处理好的、不包含重复项的序列的最后一个元素
     * 3. 快指针 fast：用于扫描数组，寻找下一个不重复的元素
     * 4. 只有当 fast 指向的元素与 slow 指向的元素不同时，才将 slow 前移并更新其值
     * <p>
     * 核心逻辑：由于数组是有序的，重复的元素必然相邻。通过跳过所有与 slow 指向元素相同的 fast 指向元素，
     * 我们可以有效地“删除”重复项，只需要将发现的新元素覆盖到数组前方即可。
     * <p>
     * 时间复杂度：O(n)，其中 n 是数组的长度，只需一次遍历
     * 空间复杂度：O(1)，原地修改数组，不需要额外空间
     *
     * @param nums 输入的有序整数数组
     * @return 修改后不含重复项的数组长度
     */
    public int removeDuplicates(int[] nums) {
        // 边界情况：如果数组为空，返回长度0
        if (nums.length == 0) return 0;

        // 初始化快慢指针：初始都指向数组的第一个元素
        int slow = 0, fast = 0;

        // 快指针遍历整个数组
        while (fast < nums.length) {
            // 如果快指针指向的元素与慢指针指向的元素不相等
            // 说明找到了一个新的唯一元素
            if (nums[slow] != nums[fast]) {
                // 先将慢指针前移一位，腾出位置存放新找到的唯一元素
                slow++;
                // 将快指针指向的新元素复制到慢指针的位置
                // 这样数组的前 slow+1 个元素就是目前发现的所有唯一元素
                nums[slow] = nums[fast];
            }
            // 无论是否找到新元素，快指针始终向后移动
            fast++;
        }

        // 最终返回不重复元素的总数
        // 由于 slow 是索引（从0开始），所以长度应该是 slow + 1
        return slow + 1;
    }

    /**
     * 删除有序链表中的重复元素，使每个元素只出现一次
     * <p>
     * 算法思路：
     * 1. 使用快慢指针（双指针）技巧，思路类似于删除有序数组中的重复项
     * 2. 慢指针 slow：指向当前已经处理好的、不含重复元素的链表末尾
     * 3. 快指针 fast：用于遍历整个链表，寻找下一个不重复的元素
     * 4. 当 fast.val != slow.val 时，说明找到了一个新的不重复元素，将其接到 slow 后面
     * <p>
     * 核心逻辑：由于链表是有序的，相同的元素必然相邻。我们通过快指针跳过重复的值，
     * 找到新值后更新慢指针并修改其值（或调整指针指向）。
     * <p>
     * 时间复杂度：O(n)，其中 n 是链表中的节点数
     * 空间复杂度：O(1)，原地修改链表节点的值
     *
     * @param head 有序链表的头节点
     * @return 处理后不含重复元素的链表头节点
     */
    public ListNode deleteDuplicates2(ListNode head) {
        // 边界情况：如果链表为空，直接返回 null
        if (head == null) return null;

        // 初始化快慢指针，初始都指向头节点
        ListNode fast = head;
        ListNode slow = head;

        // 快指针遍历整个链表
        while (fast != null) {
            // 如果快指针指向的值与慢指针指向的值不等
            // 说明快指针找到了一个新的唯一元素
            if (fast.val != slow.val) {
                // 慢指针向后移动一位
                slow = slow.next;
                // 将慢指针当前节点的值更新为快指针发现的新值
                // 注意：这里是原地修改值，这种做法在某些情况下更高效
                slow.val = fast.val;
            }
            // 快指针始终向后移动，探索新节点
            fast = fast.next;
        }

        // 遍历结束后，从 slow 之后的所有重复节点都需要断开
        // 这样 slow 之后的部分就被“逻辑删除”了
        slow.next = null;

        // 返回修改后的原链表头节点
        return head;
    }

    /**
     * 移除数组中所有等于 val 的元素，并返回移除后数组的新长度
     * <p>
     * 算法思路：
     * 1. 使用快慢指针（双指针）技巧，原地修改数组
     * 2. 快指针 fast：用于遍历整个数组，寻找不等于目标值 val 的元素
     * 3. 慢指针 slow：指向下一个即将存放“有效”元素的位置
     * 4. 只有当 fast 指向的元素不等于 val 时，才将其复制到 slow 的位置，并让 slow 前移
     * <p>
     * 核心逻辑：不需要真正“删除”元素，只需要将不等于 val 的元素依次移动到数组的前部。
     * 遍历结束后，数组的前 slow 个元素就是所有不等于 val 的元素。
     * <p>
     * 时间复杂度：O(n)，其中 n 是数组的长度，只需一次遍历
     * 空间复杂度：O(1)，原地修改数组，不需要额外空间
     *
     * @param nums 输入的整数数组
     * @param val  需要移除的目标值
     * @return 移除指定元素后数组的新长度
     */
    public int removeElement(int[] nums, int val) {
        // 初始化快慢指针，都从数组起始位置开始
        int slow = 0, fast = 0;

        // 快指针遍历整个数组
        while (fast < nums.length) {
            // 如果当前元素不等于目标值 val
            if (nums[fast] != val) {
                // 将快指针指向的有效元素移动到慢指针所在的位置
                nums[slow] = nums[fast];
                // 慢指针向前移动，准备接收下一个有效元素
                slow++;
            }
            // 快指针始终向后移动，探索数组中的每一个元素
            fast++;
        }

        // 最终 slow 指针的值就是不等于 val 的元素个数，即新数组的长度
        return slow;
    }

    /**
     * 移动零：将数组中所有的 0 移动到数组的末尾，同时保持非零元素的相对顺序
     * <p>
     * 算法思路：
     * 1. 复用 removeElement 方法：先将数组中所有不等于 0 的元素移动到数组前面
     * 2. 填充零：将剩余的位置全部填充为 0
     * <p>
     * 时间复杂度：O(n)，其中 n 是数组长度
     * 空间复杂度：O(1)，原地修改
     *
     * @param nums 输入的整数数组
     */
    public void moveZeroes(int[] nums) {
        // 步骤1：利用已有的 removeElement 方法，将所有非 0 元素移到数组前方
        // p 为处理后非 0 元素的个数，也就是下一个应该填入 0 的起始索引
        int p = removeElement(nums, 0);

        // 步骤2：使用 while 循环将数组剩余部分（索引从 p 到末尾）全部赋值为 0
        // 这样就实现了将所有 0 移动到末尾的效果
        while (p < nums.length) {
            nums[p] = 0; // 将当前位置设为 0
            p++;         // 指针向后移动
        }
    }

    /**
     * 在有序数组中寻找两个数，使其和等于目标值 target
     * <p>
     * 算法思路：
     * 1. 利用数组有序的特性，使用双指针技巧（左右指针）
     * 2. 初始化左指针 left 指向数组起始位置，右指针 right 指向数组末尾
     * 3. 在循环中计算两数之和 sum = numbers[left] + numbers[right]
     * 4. 根据 sum 与 target 的比较结果移动指针：
     * - 如果 sum == target：找到答案，返回索引（题目要求从1开始计数，故加1）
     * - 如果 sum > target：说明和太大，需要减小和，因此将右指针向左移动（right--）
     * - 如果 sum < target：说明和太小，需要增大和，因此将左指针向右移动（left++）
     * <p>
     * 时间复杂度：O(n)，其中 n 是数组的长度，最多遍历一次数组
     * 空间复杂度：O(1)，只使用了常数个额外变量
     *
     * @param numbers 输入的升序排列的整数数组
     * @param target  目标和
     * @return 包含两个数索引的数组（索引从1开始计数），若未找到则返回 [-1, -1]
     */
    public int[] twoSum(int[] numbers, int target) {
        // 初始化左右指针，分别指向数组的两端
        int left = 0, right = numbers.length - 1;

        // 当左指针小于右指针时进行循环
        while (left < right) {
            // 计算当前左右指针所指元素的和
            int sum = numbers[left] + numbers[right];

            // 情况1：找到目标和
            if (sum == target) {
                // 根据题目要求，返回从1开始计数的索引数组
                return new int[]{left + 1, right + 1};
            }
            // 情况2：当前和大于目标值，说明右边的数太大了
            else if (sum > target) {
                // 右指针左移，尝试更小的数
                right--;
            }
            // 情况3：当前和小于目标值，说明左边的数太小了
            else {
                // 左指针右移，尝试更大的数
                left++;
            }
        }

        // 如果循环结束仍未找到，返回表示未找到的数组
        return new int[]{-1, -1};
    }

    /**
     * 寻找字符串中的最长回文子串
     * <p>
     * 算法思路：中心扩展法
     * 1. 遍历字符串中的每个字符，将其作为回文串的“中心”
     * 2. 中心可以是单个字符（奇数长度回文串），也可以是两个字符之间的空隙（偶数长度回文串）
     * 3. 对于每个中心，向两边扩展，直到不再满足回文条件
     * 4. 记录并更新遍历过程中发现的最长回文子串
     * <p>
     * 复杂度分析：
     * 时间复杂度：O(n²)，其中n是字符串长度。遍历中心需要O(n)，每个中心扩展需要O(n)。
     * 空间复杂度：O(1)，只需要常数空间。
     *
     * @param s 输入字符串
     * @return 最长回文子串
     */
    public String longestPalindrome(String s) {
        String res = ""; // 用于存储当前找到的最长回文子串
        for (int i = 0; i < s.length(); i++) {
            // 以 s[i] 为中心寻找奇数长度的回文串
            // 例如 "aba"，中心是 'b' (索引1)
            String s1 = palindrome(s, i, i);

            // 以 s[i] 和 s[i+1] 之间的间隙为中心寻找偶数长度的回文串
            // 例如 "abba"，中心是 "bb" 之间的位置 (索引1和2)
            String s2 = palindrome(s, i, i + 1);

            // 如果找到的奇数长度回文串比当前最长的更长，更新结果
            res = res.length() > s1.length() ? res : s1;
            // 如果找到的偶数长度回文串比当前最长的更长，更新结果
            res = res.length() > s2.length() ? res : s2;
        }
        return res; // 返回最终的最长回文子串
    }

    /**
     * 从给定的中心向两边扩展，寻找最长的回文子串
     * <p>
     * 逻辑说明：
     * 当字符 s[l] == s[r] 时，说明 [l, r] 范围内的子串是回文的，继续向外扩展。
     * 循环停止时，s[l] != s[r] 或指针越界，此时实际的回文范围是 (l, r)，即 [l+1, r-1]。
     *
     * @param s 原始字符串
     * @param l 左指针，起始扩展位置
     * @param r 右指针，起始扩展位置
     * @return 找到的回文子串
     */
    private String palindrome(String s, int l, int r) {
        // 当指针不越界且左右字符相等时，继续向两边扩展
        while (l >= 0 && r < s.length() && s.charAt(l) == s.charAt(r)) {
            l--; // 左指针左移
            r++; // 右指针右移
        }
        // substring 是左闭右开区间 [l+1, r)
        // 因为 while 循环最后一次执行了 l-- 和 r++，导致此时 s[l] != s[r]
        // 所以有效回文子串的起始位置是 l+1，结束位置是 r-1
        return s.substring(l + 1, r);
    }
}

/**
 * 使用前缀和数组实现的数组区间求和类
 * <p>
 * 算法思路：
 * 1. 预处理：构建前缀和数组preSum，其中preSum[i]表示原数组nums[0]到nums[i-1]的累加和
 * 2. 查询：利用前缀和的性质，区间[left, right]的和 = preSum[right+1] - preSum[left]
 * <img src="https://hanserwei-1308845726.cos.ap-chengdu.myqcloud.com/markdown/20260110092935960.png" alt="前缀和示意图">
 * <p>
 * 时间复杂度：
 * - 构造函数：O(n)，其中n是输入数组长度
 * - sumRange查询：O(1)
 * 空间复杂度：O(n)，用于存储前缀和数组
 */
class NumArray {

    /**
     * 前缀和数组，长度为原数组长度+1
     * preSum[i]表示nums[0]到nums[i-1]的累加和
     * 特别地，preSum[0] = 0（空数组的和为0）
     */
    private final int[] preSum;

    /**
     * 构造函数，初始化前缀和数组
     *
     * @param nums 输入的整数数组
     */
    public NumArray(int[] nums) {
        // 前缀和数组长度比原数组多1，便于处理边界情况
        preSum = new int[nums.length + 1];

        // 构建前缀和数组
        // preSum[i] = preSum[i-1] + nums[i-1]
        // 即：前i个元素的和 = 前i-1个元素的和 + 第i个元素
        for (int i = 1; i < nums.length + 1; i++) {
            preSum[i] = preSum[i - 1] + nums[i - 1];
        }
    }

    /**
     * 查询闭区间 [left, right] 的累加和
     * <p>
     * 利用前缀和的性质：
     * nums[left] + nums[left+1] + ... + nums[right]
     * = (nums[0] + ... + nums[right]) - (nums[0] + ... + nums[left-1])
     * = preSum[right+1] - preSum[left]
     *
     * @param left  区间左端点（包含）
     * @param right 区间右端点（包含）
     * @return 闭区间 [left, right] 的累加和
     */
    public int sumRange(int left, int right) {
        // 根据前缀和公式计算区间和
        // preSum[right+1]表示nums[0]到nums[right]的和
        // preSum[left]表示nums[0]到nums[left-1]的和
        // 两者相减即为nums[left]到nums[right]的和
        return preSum[right + 1] - preSum[left];
    }
}

/**
 * 使用二维前缀和数组实现的矩阵区间求和类
 * <p>
 * 算法思路：
 * 1. 预处理：构建二维前缀和数组preSum，其中preSum[i][j]表示从矩阵左上角(0,0)到(i-1,j-1)的矩形区域内所有元素的累加和
 * 2. 查询：利用二维前缀和的性质，计算任意矩形区域[row1, col1]到[row2, col2]的和
 * <p>
 * 时间复杂度：
 * - 构造函数：O(m*n)，其中m和n是矩阵的行数和列数
 * - sumRegion查询：O(1)
 * 空间复杂度：O(m*n)，用于存储前缀和数组
 */
class NumMatrix {
    /**
     * 二维前缀和数组，大小为(m+1) x (n+1)，其中m和n是原矩阵的行数和列数
     * preSum[i][j]表示从原矩阵左上角(0,0)到右下角(i-1,j-1)的矩形区域内所有元素的累加和
     * 特别地，preSum[0][j] = 0 和 preSum[i][0] = 0（边界值为0）
     */
    private final int[][] preSum;

    /**
     * 构造函数，初始化二维前缀和数组
     *
     * @param matrix 输入的二维整数矩阵
     */
    public NumMatrix(int[][] matrix) {
        int m = matrix.length;        // 矩阵行数
        int n = matrix[0].length;     // 矩阵列数
        // 创建(m+1) x (n+1)的前缀和数组，多一行一列便于处理边界情况
        preSum = new int[m + 1][n + 1];

        // 构建二维前缀和数组
        // 对于每个位置(i,j)，计算从(0,0)到(i-1,j-1)的矩形区域内所有元素的和
        // 这样写不仅好理解，而且以后不管矩阵大小怎么变，都不会越界
        for (int i = 1; i < preSum.length; i++) {
            for (int j = 1; j < preSum[0].length; j++) {
                // 注意：这里取原矩阵 matrix 的值时，依然要 -1
                // 因为 preSum 的索引 i 对应 matrix 的 i-1
                preSum[i][j] = preSum[i - 1][j] + preSum[i][j - 1] - preSum[i - 1][j - 1]
                        + matrix[i - 1][j - 1];
            }
        }
    }

    /**
     * 查询指定矩形区域的累加和
     * <p>
     * 利用二维前缀和的性质：
     * 要计算从(row1, col1)到(row2, col2)的矩形区域的和，
     * 等于从(0,0)到(row2, col2)的大矩形的和
     * 减去从(0,0)到(row1-1, col2)的上方矩形的和
     * 减去从(0,0)到(row2, col1-1)的左方矩形的和
     * 加上从(0,0)到(row1-1, col1-1)的左上角重叠矩形的和（因为被减了两次）
     * <img src="https://hanserwei-1308845726.cos.ap-chengdu.myqcloud.com/markdown/20260109203824773.png" alt="二维前缀和示意图">
     *
     * @param row1 矩形区域的起始行索引（包含）
     * @param col1 矩形区域的起始列索引（包含）
     * @param row2 矩形区域的结束行索引（包含）
     * @param col2 矩形区域的结束列索引（包含）
     * @return 指定矩形区域的累加和
     */
    public int sumRegion(int row1, int col1, int row2, int col2) {
        // 根据二维前缀和公式计算矩形区域和
        // preSum[row2+1][col2+1]: 从(0,0)到(row2,col2)的矩形和
        // preSum[row1][col2+1]: 从(0,0)到(row1-1,col2)的矩形和（上方部分）
        // preSum[row2+1][col1]: 从(0,0)到(row2,col1-1)的矩形和（左方部分）
        // preSum[row1][col1]: 从(0,0)到(row1-1,col1-1)的矩形和（左上角重叠部分）
        return preSum[row2 + 1][col2 + 1] - preSum[row1][col2 + 1] - preSum[row2 + 1][col1] + preSum[row1][col1];
    }
}

class ListNode {
    int val;
    ListNode next;

    ListNode(int val) {
        this.val = val;
    }

}

/**
 * 设计一个支持在末尾添加数字并计算最后 k 个数字乘积的数据结构
 * <p>
 * 算法思路：
 * 1. 使用前缀乘积数组：维护一个数组记录从开头到当前位置所有数字的累积乘积
 * 2. 当遇到0时：清空前缀乘积数组，重新开始（因为任何数乘0都为0）
 * 3. 查询乘积：利用前缀乘积的除法性质快速计算最后k个数的乘积
 * <p>
 * 时间复杂度：
 * - add操作：O(1)
 * - getProduct操作：O(1)
 * 空间复杂度：O(n)，其中n是add操作的次数
 */
class ProductOfNumbers {

    /**
     * 前缀乘积数组，preProduct[i]表示从第1个到第i个添加的数字的乘积
     * 初始时添加1作为哨兵值，方便后续计算
     * 当遇到0时，清空整个数组并重新添加1，因为0会使后续所有乘积变为0
     */
    List<Integer> preProduct = new ArrayList<>();

    /**
     * 构造函数：初始化前缀乘积数组，在开头放置哨兵值1
     * 哨兵值的作用是简化边界处理，使得getProduct(k)可以直接使用除法计算
     */
    public ProductOfNumbers() {
        preProduct.add(1);
    }

    /**
     * 在数据流末尾添加一个数字
     * <p>
     * 算法步骤：
     * 1. 如果添加的是0，清空前缀乘积数组并重新添加哨兵值1
     * 2. 否则，计算当前前缀乘积并添加到数组末尾
     *
     * @param num 要添加的数字
     */
    public void add(int num) {
        // 特殊处理：如果添加的数字是0
        // 因为任何数乘以0都为0，且0会影响之后所有的乘积计算
        // 所以直接清空前缀乘积数组，重新开始计算
        if (num == 0) {
            preProduct.clear();      // 清空之前的前缀乘积
            preProduct.add(1);       // 重新添加哨兵值1
            return;                  // 直接返回，不再执行后续逻辑
        }

        // 获取当前前缀乘积数组的长度
        int size = preProduct.size();

        // 计算新的前缀乘积：前一个前缀乘积 × 当前数字
        // preProduct.get(size - 1)是前一个前缀乘积
        // num是要添加的新数字
        preProduct.add(preProduct.get(size - 1) * num);
    }

    /**
     * 计算最后 k 个数字的乘积
     * <p>
     * 算法原理：
     * 利用前缀乘积数组的性质：
     * 如果要求第(n-k+1)个到第n个数字的乘积
     * 等于前n个数字的乘积 ÷ 前(n-k)个数字的乘积
     * 即：preProduct[n] / preProduct[n-k]
     *
     * @param k 需要计算乘积的数字个数（从末尾开始计数）
     * @return 最后 k 个数字的乘积
     */
    public int getProduct(int k) {
        // 获取前缀乘积数组的长度
        int size = preProduct.size();

        // 边界检查：如果请求的k大于已有的数字个数（不包括哨兵）
        // 说明在最近的k个数字中一定包含0（因为遇到0时会清空数组）
        // 所以返回0
        if (size - 1 < k) {  // size-1是因为第一个元素是哨兵值
            return 0;
        }

        // 计算最后k个数字的乘积
        // preProduct.get(size - 1): 前面所有数字的乘积（包括最新添加的）
        // preProduct.get(size - k - 1): 前面(size - k - 1)个数字的乘积
        // 两者相除即为最后k个数字的乘积
        // 例如：要求最后3个数的乘积，就是总乘积 ÷ 前面较老的部分的乘积
        return preProduct.get(size - 1) / preProduct.get(size - k - 1);
    }
}

/**
 * 差分数组工具类
 * <p>
 * 算法思路：
 * 1. 差分数组：构造一个数组diff，其中diff[i] = nums[i] - nums[i-1]（i>0），diff[0] = nums[0]
 * 2. 区间修改：要在区间[i,j]上增加val，只需diff[i] += val，diff[j+1] -= val（如果j+1<n）
 * 3. 还原数组：通过前缀和还原原数组，res[i] = res[i-1] + diff[i]
 * <p>
 * 应用场景：频繁对数组的某个区间进行增减操作，然后还原数组
 * 时间复杂度：构造O(n)，区间修改O(1)，还原数组O(n)
 */
class Difference {
    /**
     * 差分数组：diff[i]表示原数组nums[i]与nums[i-1]的差值
     * 特别地，diff[0] = nums[0]
     * 通过差分数组可以高效地进行区间增减操作
     */
    private final int[] diff;

    /**
     * 构造差分数组
     * <p>
     * 初始化过程：
     * 1. diff[0] = nums[0]（第一个元素的差值就是它本身）
     * 2. 对于i > 0，diff[i] = nums[i] - nums[i-1]（相邻元素的差值）
     *
     * @param nums 输入的原始数组
     */
    public Difference(int[] nums) {
        // 参数校验：确保输入数组不为null且长度大于0
        assert nums != null && nums.length > 0;

        // 初始化差分数组，长度与原数组相同
        diff = new int[nums.length];

        // 设置差分数组的第一个元素等于原数组的第一个元素
        diff[0] = nums[0];

        // 构建差分数组：每个位置存储相邻元素的差值
        for (int i = 1; i < nums.length; i++) {
            // 当前位置的差值 = 当前元素 - 前一个元素
            diff[i] = nums[i] - nums[i - 1];
        }
    }

    /**
     * 对区间[i, j]增加指定值val
     * <p>
     * 核心思想：利用差分数组的特性，O(1)时间完成区间修改
     * 1. diff[i] += val：影响从位置i开始的所有元素
     * 2. diff[j + 1] -= val：抵消从位置j+1开始的影响（如果存在）
     * <p>
     * 举例：原数组[1,2,3,4,5]，对区间[1,3]增加2
     * - 操作前：diff = [1,1,1,1,1]
     * - 操作后：diff[1] += 2, diff[4] -= 2 → diff = [1,3,1,1,-1]
     * - 还原后：[1,4,5,6,5]（位置1-3确实增加了2）
     *
     * @param i   区间起始位置（包含）
     * @param j   区间结束位置（包含）
     * @param val 要增加的值
     */
    public void increment(int i, int j, int val) {
        // 在起始位置增加val，这样从i开始的所有元素都会受到影响
        diff[i] += val;

        // 如果结束位置的下一个位置存在，则减少val，抵消后续的影响
        // 这样只有[i, j]区间内的元素增加了val
        if (j + 1 < diff.length) {
            // 在j+1位置减少val，使得从j+1开始的元素不受此次增量影响
            diff[j + 1] -= val;
        }
    }

    /**
     * 还原原始数组
     * <p>
     * 通过前缀和的方式从差分数组还原出修改后的原数组
     * 1. res[0] = diff[0]
     * 2. 对于i > 0，res[i] = res[i-1] + diff[i]
     * <p>
     * 这是因为：如果diff是原数组的差分数组，则原数组是diff的前缀和数组
     *
     * @return 还原后的数组
     */
    public int[] result() {
        // 创建结果数组，长度与差分数组相同
        int[] res = new int[diff.length];

        // 第一个元素直接等于差分数组的第一个元素
        res[0] = diff[0];

        // 通过前缀和还原数组：每个元素等于前一个元素加上对应的差值
        for (int i = 1; i < diff.length; i++) {
            // 当前元素 = 前一个元素 + 当前位置的差值
            res[i] = res[i - 1] + diff[i];
        }

        return res; // 返回还原后的数组
    }
}