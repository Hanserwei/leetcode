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
     * 翻转 StringBuilder 中指定区间的字符
     * <p>
     * 算法思路：
     * 使用双指针从两端向中间移动，交换对应位置的字符
     * <p>
     * 示例：
     * 输入: "hello", left=0, right=4
     * 过程: h<->o, e<->l, l（中心不动）
     * 输出: "olleh"
     *
     * @param sb    要操作的 StringBuilder 对象
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
     * 翻转 StringBuilder 中的每个单词
     * <p>
     * 算法步骤：
     * 1. 遍历字符串，找到每个单词的起始和结束位置
     * 2. 对每个单词调用reverse方法进行翻转
     * 3. 更新指针继续查找下一个单词
     * <p>
     * 单词识别：以空格为分隔符，连续的非空格字符组成一个单词
     *
     * @param sb 要操作的 StringBuilder 对象
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

    /**
     * 判断 s2 是否包含 s1 的排列
     * <p>
     * 算法思路：
     * 1. 使用滑动窗口技术，窗口大小固定为 s1 的长度
     * 2. 维护两个哈希表：need 记录 s1 中每个字符的需求频次，window 记录当前窗口中各字符的实际频次
     * 3. 当窗口大小等于 s1 长度时，检查窗口中的字符频次是否与 s1 中的字符频次完全匹配
     * 4. 如果匹配成功，说明找到了 s1 的一个排列；否则继续滑动窗口
     * <p>
     * 时间复杂度：O(n)，其中 n 是 s2 的长度
     * 空间复杂度：O(1)，因为字符集大小是常数（最多26个小写字母）
     *
     * @param s1 目标字符串，我们要寻找其排列
     * @param s2 源字符串，在其中查找 s1 的排列
     * @return 如果 s2 包含 s1 的某个排列，返回 true；否则返回 false
     */
    public boolean checkInclusion(String s1, String s2) {
        // characterNeedCount：记录s1中每个字符的需求频次
        Map<Character, Integer> characterNeedCount = new HashMap<>();
        // windowCharacterCount：记录当前滑动窗口中各字符的实际频次
        Map<Character, Integer> windowCharacterCount = new HashMap<>();

        // 统计s1中每个字符的出现次数，构建需求映射
        for (char character : s1.toCharArray()) {
            characterNeedCount.put(character, characterNeedCount.getOrDefault(character, 0) + 1);
        }

        // 定义滑动窗口的左右边界
        int windowLeft = 0;
        int windowRight = 0;

        // matchedCharacterTypes：记录当前窗口中满足需求频次的字符种类数
        int matchedCharacterTypes = 0;

        // 开始滑动窗口遍历s2
        while (windowRight < s2.length()) {
            // 扩展窗口：将右边界字符纳入窗口
            char incomingChar = s2.charAt(windowRight);
            windowRight++;

            // 如果当前字符是s1中需要的字符
            if (characterNeedCount.containsKey(incomingChar)) {
                // 更新窗口中该字符的计数
                windowCharacterCount.put(incomingChar, windowCharacterCount.getOrDefault(incomingChar, 0) + 1);

                // 如果窗口中该字符的频次恰好等于需求频次，说明该字符种类达标
                if (windowCharacterCount.get(incomingChar).equals(characterNeedCount.get(incomingChar))) {
                    matchedCharacterTypes++;
                }
            }

            // 收缩窗口：当窗口长度达到s1长度时开始收缩（因为排列的长度必须等于s1的长度）
            while (windowRight - windowLeft >= s1.length()) {
                // 检查当前窗口是否满足条件：所有字符种类都满足需求频次
                if (matchedCharacterTypes == characterNeedCount.size()) {
                    return true; // 找到了s1的一个排列
                }

                // 收缩窗口：将左边界字符移出窗口
                char outgoingChar = s2.charAt(windowLeft);
                windowLeft++;

                // 如果移出的字符是s1中需要的字符
                if (characterNeedCount.containsKey(outgoingChar)) {
                    // 如果移出前该字符的频次恰好等于需求频次，移出后就不满足了，matchedCharacterTypes减1
                    if (windowCharacterCount.get(outgoingChar).equals(characterNeedCount.get(outgoingChar))) {
                        matchedCharacterTypes--;
                    }
                    // 更新窗口中该字符的计数
                    windowCharacterCount.put(outgoingChar, windowCharacterCount.get(outgoingChar) - 1);
                }
            }
        }
        // 遍历完整个s2都没找到s1的排列，返回false
        return false;
    }

    /**
     * 找到字符串 s 中所有与字符串 t 互为字母异位词的子串的起始索引
     * <p>
     * 算法思路：
     * 1. 使用滑动窗口技术，窗口大小固定为 t 的长度
     * 2. 维护两个哈希表：need 记录 t 中每个字符的需求频次，window 记录当前窗口中各字符的实际频次
     * 3. 当窗口大小等于 t 长度时，检查窗口中的字符频次是否与 t 中的字符频次完全匹配
     * 4. 如果匹配成功，记录起始索引；否则继续滑动窗口
     * <p>
     * 字母异位词定义：两个字符串包含相同的字母，且每个字母出现的频次相同，但顺序可以不同
     * <p>
     * 时间复杂度：O(n)，其中 n 是 s 的长度
     * 空间复杂度：O(1)，因为字符集大小是常数（最多26个小写字母）
     *
     * @param s 源字符串，在其中查找异位词
     * @param t 目标字符串，我们要寻找其异位词
     * @return 所有异位词的起始索引列表
     */
    public List<Integer> findAnagrams(String s, String t) {
        // need：记录目标字符串t中每个字符的需求频次
        Map<Character, Integer> need = new HashMap<>();
        // window：记录当前滑动窗口中各字符的实际频次
        Map<Character, Integer> window = new HashMap<>();

        // 统计目标字符串t中每个字符的出现次数，构建需求映射
        for (char c : t.toCharArray()) {
            need.put(c, need.getOrDefault(c, 0) + 1);
        }

        // 定义滑动窗口的左右边界
        int left = 0, right = 0;
        // 存储所有异位词的起始索引
        List<Integer> ans = new ArrayList<>();
        // valid：记录当前窗口中满足需求频次的字符种类数
        int valid = 0;

        // 开始滑动窗口遍历源字符串s
        while (right < s.length()) {
            // 扩展窗口：将右边界字符纳入窗口
            char c = s.charAt(right);
            right++;

            // 如果当前字符是目标字符串t中需要的字符
            if (need.containsKey(c)) {
                // 更新窗口中该字符的计数
                window.put(c, window.getOrDefault(c, 0) + 1);

                // 如果窗口中该字符的频次恰好等于需求频次，说明该字符种类达标
                if (window.get(c).equals(need.get(c))) {
                    valid++;
                }
            }

            // 收缩窗口：当窗口长度达到目标字符串t的长度时开始收缩
            // 因为异位词的长度必须等于目标字符串t的长度
            while (right - left >= t.length()) {
                // 检查当前窗口是否满足条件：所有字符种类都满足需求频次
                // 即窗口中的字符频次与目标字符串t的字符频次完全一致
                if (valid == need.size()) {
                    // 找到了一个异位词，记录起始索引
                    ans.add(left);
                }

                // 收缩窗口：将左边界字符移出窗口
                char d = s.charAt(left);
                left++;

                // 如果移出的字符是目标字符串t中需要的字符
                if (need.containsKey(d)) {
                    // 如果移出前该字符的频次恰好等于需求频次，移出后就不满足了，valid减1
                    if (window.get(d).equals(need.get(d))) {
                        valid--;
                    }
                    // 更新窗口中该字符的计数
                    window.put(d, window.get(d) - 1);
                }
            }
        }
        // 返回所有找到的异位词的起始索引列表
        return ans;
    }

    /**
     * 计算字符串中最长无重复字符子串的长度
     * <p>
     * 算法思路：滑动窗口（双指针）+ 哈希表
     * 1. 使用左右双指针维护一个滑动窗口，窗口内不包含重复字符
     * 2. 使用哈希表记录窗口中每个字符的出现次数
     * 3. 右指针不断扩展窗口，当遇到重复字符时收缩窗口
     * 4. 实时更新最长子串长度
     * <p>
     * 时间复杂度：O(n)，其中n是字符串长度，每个字符最多被访问两次（一次入窗，一次出窗）
     * 空间复杂度：O(min(m,n))，其中m是字符集大小，哈希表最多存储min(m,n)个字符
     *
     * @param s 输入字符串
     * @return 最长无重复字符子串的长度
     */
    public int lengthOfLongestSubstring(String s) {
        // 创建哈希表存储滑动窗口中每个字符的出现次数
        // key: 字符，value: 该字符在窗口中的出现次数
        Map<Character, Integer> window = new HashMap<>();

        // 右指针：用于扩展窗口的右边界
        int right = 0;
        // 记录最长无重复子串的长度
        int ans = 0;
        // 左指针：用于收缩窗口的左边界
        int left = 0;

        // 当右指针未到达字符串末尾时继续扩展窗口
        while (right < s.length()) {
            // 获取右指针指向的字符
            char c = s.charAt(right);
            // 右指针右移，扩展窗口右边界
            right++;
            // 将字符c加入窗口，更新其出现次数
            window.put(c, window.getOrDefault(c, 0) + 1);

            // 当窗口中字符c的出现次数超过1（即出现重复）时，需要收缩窗口
            while (window.get(c) > 1) {
                // 获取左指针指向的字符
                char d = s.charAt(left);
                // 左指针右移，收缩窗口左边界
                left++;
                // 将字符d从窗口中移除，更新其出现次数
                window.put(d, window.get(d) - 1);
            }

            // 此时窗口[left, right)中不包含重复字符，更新最长子串长度
            // right - left 是当前窗口的长度
            ans = Math.max(ans, right - left);
        }

        // 返回最长无重复字符子串的长度
        return ans;
    }

    /**
     * 通过移除数组两端元素使剩余元素和等于x的最少操作次数
     * <p>
     * 算法思路：
     * 1. 转换思维：不是直接找要移除的元素，而是找中间连续子数组的和等于 sum - x
     * 2. 使用滑动窗口：找到和为 target = sum - x 的最长子数组
     * 3. 最少操作数 = 总长度 - 最长子数组长度
     * <p>
     * 核心思想：如果整个数组的和是sum，我们要移除两端的元素使剩余和为x，
     * 那么中间保留的连续子数组的和应该是sum-x。为了让移除操作最少，
     * 我们需要保留尽可能长的中间子数组，这样移除的元素就最少。
     * <p>
     * 时间复杂度：O(n)，其中n是数组长度
     * 空间复杂度：O(1)，只使用了常数个额外变量
     *
     * @param nums 输入的正整数数组
     * @param x    目标剩余和
     * @return 最少操作次数，如果无法达到目标则返回-1
     */
    public int minOperations(int[] nums, int x) {
        int n = nums.length; // 获取数组长度
        int left = 0;        // 滑动窗口左指针
        int right = 0;       // 滑动窗口右指针

        // 计算整个数组的总和
        int sum = 0;
        for (int i : nums) {
            sum += i;
        }

        // 目标值：中间连续子数组的和应该等于 sum - x
        // 这样两端移除的元素和就等于 x
        int target = sum - x;

        int windowSum = 0;      // 当前滑动窗口的和
        int maxLen = Integer.MIN_VALUE; // 记录满足条件的最长子数组长度

        // 使用滑动窗口寻找和为 target 的最长子数组
        while (right < n) {
            // 扩展窗口：将右指针指向的元素加入窗口
            windowSum += nums[right++];

            // 收缩窗口：如果窗口和大于目标值，移动左指针
            while (windowSum > target && left < right) {
                windowSum -= nums[left++]; // 从窗口中移除左指针元素
            }

            // 检查当前窗口是否满足条件：和等于目标值
            if (windowSum == target) {
                // 更新最长子数组长度
                maxLen = Math.max(maxLen, right - left);
            }
        }

        // 如果找到了满足条件的子数组，返回需要移除的元素个数
        // 否则返回-1表示无法达到目标
        return maxLen == Integer.MIN_VALUE ? -1 : n - maxLen;
    }

    /**
     * 计算数组中所有乘积小于 k 的连续子数组的个数
     * <p>
     * 算法思路：滑动窗口（双指针）+ 乘积优化
     * 1. 使用左右双指针维护一个滑动窗口，窗口内元素的乘积小于k
     * 2. 右指针不断扩展窗口，当窗口内乘积大于等于k时收缩窗口
     * 3. 对于每个右端点，统计以该位置为结尾的所有满足条件的子数组个数
     * <p>
     * 核心思想：对于当前右指针位置，所有以该位置为结尾且乘积小于k的子数组
     * 的数量等于当前窗口的长度（right - left）。这是因为窗口[left, right)
     * 内的所有子数组[nums[left], nums[right]], [nums[left+1], nums[right]], ...
     * [nums[right-1], nums[right]], [nums[right]]都满足条件。
     * <p>
     * 时间复杂度：O(n)，其中n是数组长度，每个元素最多被访问两次（一次入窗，一次出窗）
     * 空间复杂度：O(1)，只使用了常数个额外变量
     *
     * @param nums 输入的正整数数组
     * @param k    目标乘积阈值
     * @return 乘积小于k的连续子数组的个数
     */
    public int numSubarrayProductLessThanK(int[] nums, int k) {
        int n = nums.length;         // 获取数组长度
        int left = 0;                // 滑动窗口左指针，指向窗口的起始位置
        int right = 0;               // 滑动窗口右指针，指向窗口的结束位置的下一个

        int products = 1;            // 当前窗口内所有元素的乘积，初始为1（乘法单位元）
        int ans = 0;                 // 记录满足条件的子数组总数

        // 当右指针未到达数组末尾时继续扩展窗口
        while (right < n) {
            // 扩展窗口：将右指针指向的元素加入窗口，更新乘积
            products *= nums[right++]; // 先将nums[right]乘入products，然后右指针右移

            // 收缩窗口：如果当前窗口的乘积大于等于k，需要移动左指针缩小窗口
            // 注意：必须保证left < right，避免窗口变成负长度
            while (products >= k && left < right) {
                products /= nums[left++]; // 先将nums[left]从products中除掉，然后左指针右移
            }

            // 此时窗口[left, right)内所有元素的乘积小于k
            // 以right-1位置为结尾的所有满足条件的子数组个数为 right - left
            // 这些子数组分别是：[right-1], [right-2, right-1], ..., [left, left+1, ..., right-1]
            ans += right - left; // 累加当前右端点贡献的子数组数量
        }

        // 返回所有满足条件的连续子数组的总数
        return ans;
    }

    /**
     * 找到最长的连续子数组，使得最多翻转k个0后可以全部变成1
     * <p>
     * 算法思路：滑动窗口（双指针）+ 窗口内0的个数统计
     * 1. 使用左右双指针维护一个滑动窗口，窗口内最多包含k个0
     * 2. windowOneSum记录窗口内1的个数
     * 3. right - left - windowOneSum表示窗口内0的个数
     * 4. 当窗口内0的个数超过k时，收缩窗口
     * 5. 实时更新最长窗口长度
     * <p>
     * 核心思想：我们要找到一个最长的窗口，窗口内最多有k个0，这样我们就可以把这k个0都变成1
     * <p>
     * 时间复杂度：O(n)，其中n是数组长度，每个元素最多被访问两次（一次入窗，一次出窗）
     * 空间复杂度：O(1)，只使用了常数个额外变量
     *
     * @param nums 输入的二进制数组（只包含0和1）
     * @param k    最多可以翻转的0的个数
     * @return 最长连续子数组的长度，该子数组可以通过翻转最多k个0变成全1
     */
    public int longestOnes(int[] nums, int k) {
        // 滑动窗口左指针，指向窗口的起始位置
        int left = 0;
        // 滑动窗口右指针，指向窗口的结束位置的下一个
        int right = 0;

        // 记录当前窗口内1的个数
        int windowOneSum = 0;
        // 记录满足条件的最长子数组长度
        int res = 0;

        // 当右指针未到达数组末尾时继续扩展窗口
        while (right < nums.length) {
            // 扩展窗口：将右指针指向的元素纳入窗口
            if (nums[right] == 1) {
                // 如果当前元素是1，更新窗口内1的个数
                windowOneSum++;
            }
            // 右指针右移，扩展窗口右边界
            right++;

            // 收缩窗口：如果窗口内0的个数超过k个，需要收缩窗口
            // right - left 是当前窗口的总长度
            // windowOneSum 是当前窗口内1的个数
            // right - left - windowOneSum 就是当前窗口内0的个数
            while (right - left - windowOneSum > k) {
                // 如果左指针指向的元素是1，从窗口中移除时需要减少1的计数
                if (nums[left] == 1) {
                    windowOneSum--;
                }
                // 左指针右移，收缩窗口左边界
                left++;
            }

            // 此时窗口[left, right)内0的个数不超过k个，满足条件
            // 更新最长子数组长度
            res = Math.max(res, right - left);
        }

        // 返回最长的连续子数组长度
        return res;
    }


    /**
     * 替换字符使其连续 - LeetCode 424题
     * <p>
     * 问题描述：
     * 给定一个字符串s和一个整数k，你可以选择字符串中的任意一个字符，并将其更改为任何其他大写英文字符。
     * 该操作最多可以执行k次。在执行上述操作后，返回包含相同字母的最长子字符串的长度。
     * <p>
     * 算法思路：滑动窗口（双指针）+ 字符频次统计
     * 1. 使用左右双指针维护一个滑动窗口
     * 2. windowCharCount数组记录窗口内每个字符的出现次数
     * 3. windowMaxCount记录窗口内出现次数最多的字符的频次
     * 4. 关键判断：如果窗口长度 - 最高频字符数 > k，说明需要替换的字符超过k个，需要收缩窗口
     * 5. 窗口内其他字符的总数 = 窗口长度 - 最高频字符数
     * <p>
     * 核心思想：
     * 要使窗口内所有字符相同，最优策略是将其他字符都替换成出现次数最多的那个字符。
     * 如果窗口内除了最高频字符外的其他字符总数不超过k个，就可以通过k次替换使窗口内所有字符相同。
     * 因此，有效窗口的条件是：窗口长度 - 最高频字符数 <= k
     * <p>
     * 时间复杂度：O(n)，其中n是字符串长度，每个字符最多被访问两次（一次入窗，一次出窗）
     * 空间复杂度：O(1)，使用固定大小的数组（26个字母）
     *
     * @param s 输入的大写字母字符串
     * @param k 最多可以执行的替换次数
     * @return 执行替换后包含相同字母的最长子字符串的长度
     */
    public int characterReplacement(String s, int k) {
        // 滑动窗口左指针，指向窗口的起始位置
        int left = 0;
        // 滑动窗口右指针，指向窗口的结束位置的下一个
        int right = 0;

        // 窗口内各字符的出现次数统计数组
        // windowCharCount[i]表示字符('A' + i)在当前窗口中的出现次数
        // 例如：windowCharCount[0]表示'A'的出现次数，windowCharCount[1]表示'B'的出现次数
        int[] windowCharCount = new int[26];

        // 记录当前窗口内出现次数最多的字符的频次
        // 这个值决定了窗口的有效性：如果窗口长度 - windowMaxCount > k，说明需要收缩窗口
        int windowMaxCount = 0;

        // 记录满足条件的最长子字符串长度
        int res = 0;

        // 当右指针未到达字符串末尾时继续扩展窗口
        while (right < s.length()) {
            // 扩展窗口：将右指针指向的字符纳入窗口
            // s.charAt(right) - 'A' 将字符转换为数组索引（0-25）
            // 例如：'A' - 'A' = 0, 'B' - 'A' = 1, ..., 'Z' - 'A' = 25
            int c = s.charAt(right++) - 'A';

            // 更新窗口中字符c的出现次数
            windowCharCount[c]++;

            // 更新窗口内最高频字符的出现次数
            // 每次添加新字符后，重新计算当前窗口内的最大频次
            windowMaxCount = Math.max(windowMaxCount, windowCharCount[c]);

            // 收缩窗口：如果窗口内需要替换的字符数超过k个
            // 窗口长度 = right - left（right已经+1了，所以不需要再+1）
            // 需要替换的字符数 = 窗口长度 - 最高频字符数
            // 当需要替换的字符数 > k 时，说明无法通过k次替换使窗口内所有字符相同，需要收缩窗口
            while (right - left - windowMaxCount > k) {
                // 将左指针指向的字符移出窗口，更新其出现次数
                // s.charAt(left) - 'A' 将字符转换为数组索引
                windowCharCount[s.charAt(left++) - 'A']--;
                // 注意：这里不需要更新windowMaxCount，因为：
                // 1. 移出的字符可能不是最高频字符，windowMaxCount保持不变
                // 2. 即使移出的是最高频字符，我们只关心历史最大值，不影响结果
            }

            // 此时窗口[left, right)满足条件：需要替换的字符数 <= k
            // 更新最长子字符串长度
            // right - left 是当前窗口的长度
            res = Math.max(res, right - left);
        }

        // 返回满足条件的最长子字符串长度
        return res;
    }


    /**
     * 判断数组中是否存在两个相同元素的索引之差不超过k
     * <p>
     * 算法思路：滑动窗口（双指针）+ 哈希集合
     * 1. 使用滑动窗口维护一个长度不超过k的窗口
     * 2. 使用HashSet存储窗口内的元素，快速判断是否存在重复
     * 3. 当窗口长度超过k时，移除最左边的元素
     * 4. 如果发现重复元素，说明存在满足条件的两个索引
     * <p>
     * 核心思想：维护一个大小不超过k的滑动窗口，窗口内任意两个元素的索引之差必然不超过k
     * 如果窗口内出现重复元素，就说明存在索引之差不超过k的重复元素
     * <p>
     * 时间复杂度：O(n)，其中n是数组长度，每个元素最多被访问两次（一次入窗，一次出窗）
     * 空间复杂度：O(min(n, k))，HashSet最多存储k+1个元素
     *
     * @param nums 输入的整数数组
     * @param k    索引之差的上限
     * @return 如果存在两个相同元素的索引之差不超过k则返回true，否则返回false
     */
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        // 滑动窗口左指针，指向窗口的起始位置
        int left = 0;
        // 滑动窗口右指针，指向窗口的结束位置的下一个
        int right = 0;

        // 使用HashSet存储当前窗口内的元素
        // 利用HashSet的O(1)查找特性，快速判断元素是否存在
        Set<Integer> window = new HashSet<>();

        // 当右指针未到达数组末尾时继续扩展窗口
        while (right < nums.length) {
            // 检查当前元素是否已经在窗口中存在（即是否重复）
            // 如果存在，说明在k范围内找到了重复元素
            if (window.contains(nums[right])) {
                // 找到满足条件的重复元素，返回true
                return true;
            }

            // 将当前元素加入窗口，并将右指针右移
            // 扩展窗口右边界
            window.add(nums[right++]);

            // 收缩窗口：确保窗口长度不超过k
            // right - left 是当前窗口的长度
            // 如果窗口长度 > k，说明窗口内元素的索引之差可能超过k
            while (right - left > k) {
                // 移除左指针指向的元素，收缩窗口左边界
                // 保持窗口大小不超过k+1（索引差不超过k）
                window.remove(nums[left++]);
            }
        }

        // 遍历完整个数组都没有找到满足条件的重复元素
        return false;
    }


    /**
     * 判断数组中是否存在两个不同元素，满足索引之差不超过indexDiff且值之差不超过valueDiff
     * <p>
     * 算法思路：滑动窗口（双指针）+ TreeSet（有序集合）
     * 1. 使用滑动窗口维护一个长度不超过indexDiff的窗口
     * 2. 使用TreeSet存储窗口内的元素，利用其有序性快速查找满足值差条件的元素
     * 3. 对于每个新元素nums[right]，检查窗口中是否存在元素x，使得|x - nums[right]| <= valueDiff
     * 4. 利用TreeSet的ceiling和floor方法：
     *    - ceiling(nums[right])：返回集合中大于或等于nums[right]的最小元素
     *    - floor(nums[right])：返回集合中小于或等于nums[right]的最大元素
     * 5. 如果ceiling(nums[right]) - nums[right] <= valueDiff，说明找到了满足条件的元素
     * 6. 或者如果nums[right] - floor(nums[right]) <= valueDiff，也说明找到了满足条件的元素
     * <p>
     * 核心思想：
     * - 窗口大小限制保证索引差 <= indexDiff
     * - TreeSet的ceiling和floor方法能够O(log k)时间找到最接近nums[right]的元素
     * - 只需检查最接近的两个元素（一个大于等于，一个小于等于）即可判断是否存在值差 <= valueDiff的元素
     * <p>
     * 为什么只检查ceiling和floor：
     * 假设存在窗口内的元素x满足|x - nums[right]| <= valueDiff
     * - 如果x >= nums[right]，那么ceiling(nums[right])必定 <= x，且ceiling更接近nums[right]
     * - 如果x <= nums[right]，那么floor(nums[right])必定 >= x，且floor更接近nums[right]
     * 因此，如果ceiling和floor都不满足条件，其他元素也不可能满足条件
     * <p>
     * 时间复杂度：O(n log k)，其中n是数组长度，k是窗口大小(indexDiff+1)
     * 每个元素的插入和删除操作是O(log k)，ceiling和floor查询也是O(log k)
     * 空间复杂度：O(k)，TreeSet最多存储k+1个元素
     *
     * @param nums 输入的整数数组
     * @param indexDiff 索引之差的上限
     * @param valueDiff 值之差的上限
     * @return 如果存在满足条件的两个元素返回true，否则返回false
     */
    public boolean containsNearbyAlmostDuplicate(int[] nums, int indexDiff, int valueDiff) {
        // 滑动窗口左指针，指向窗口的起始位置
        int left = 0;
        // 滑动窗口右指针，指向窗口的结束位置的下一个
        int right = 0;

        // 使用TreeSet存储当前窗口内的元素
        // TreeSet基于红黑树实现，能够保持元素有序，并支持O(log k)的ceiling和floor查询
        TreeSet<Integer> window = new TreeSet<>();

        // 当右指针未到达数组末尾时继续扩展窗口
        while (right < nums.length) {
            // ceiling方法：返回集合中大于或等于nums[right]的最小元素
            // 如果这样的元素存在，它是窗口中最接近且不小于nums[right]的值
            Integer ceiling = window.ceiling(nums[right]);
            
            // 检查ceiling是否满足值差条件
            // ceiling - nums[right] 表示ceiling与当前元素的差值
            // 如果这个差值 <= valueDiff，说明找到了满足条件的元素
            if (ceiling != null && ceiling - nums[right] <= valueDiff) {
                // 找到满足条件的元素对，返回true
                return true;
            }

            // floor方法：返回集合中小于或等于nums[right]的最大元素
            // 如果这样的元素存在，它是窗口中最接近且不大于nums[right]的值
            Integer floor = window.floor(nums[right]);
            
            // 检查floor是否满足值差条件
            // nums[right] - floor 表示当前元素与floor的差值
            // 如果这个差值 <= valueDiff，说明找到了满足条件的元素
            if (floor != null && nums[right] - floor <= valueDiff) {
                // 找到满足条件的元素对，返回true
                return true;
            }

            // 将当前元素加入窗口，并将右指针右移
            // 扩展窗口右边界
            window.add(nums[right++]);

            // 收缩窗口：确保窗口长度不超过indexDiff
            // right - left 是当前窗口的长度
            // 如果窗口长度 > indexDiff，说明窗口内元素的索引之差可能超过indexDiff
            if (right - left > indexDiff) {
                // 移除左指针指向的元素，收缩窗口左边界
                // 保持窗口大小不超过indexDiff+1（索引差不超过indexDiff）
                window.remove(nums[left++]);
            }
        }
        
        // 遍历完整个数组都没有找到满足条件的元素对
        return false;
    }

    /**
     * 长度最小的子数组 - LeetCode 209题
     * <p>
     * 问题描述：
     * 给定一个含有 n 个正整数的数组和一个正整数 target。
     * 找出该数组中满足其和 ≥ target 的长度最小的连续子数组，并返回其长度。如果不存在符合条件的子数组，返回 0。
     * <p>
     * 算法思路：滑动窗口（双指针）
     * 1. 使用左右双指针维护一个“窗口”，窗口内的元素和为 sum。
     * 2. 右指针 right 主动向右移动，不断将元素加入窗口以增加 sum。
     * 3. 当 sum ≥ target 时，尝试收缩左指针 left，在保持 sum ≥ target 的前提下寻找最小窗口。
     * 4. 在收缩过程中，不断更新结果 res 为当前窗口的最小长度。
     * <p>
     * 核心思想：
     * 利用窗口内元素和的单调性（因为数组元素都是正整数）。
     * 窗口像一条“毛毛虫”一样在数组上爬行：右边伸长（找满足条件的情况），左边缩短（找最优解）。
     * <p>
     * 时间复杂度：O(n)，虽然有嵌套的 while 循环，但每个元素最多被 right 访问一次，被 left 访问一次。
     * 空间复杂度：O(1)，只使用了常数个额外变量。
     *
     * @param target 目标和
     * @param nums   输入正整数数组
     * @return 满足条件的最短子数组长度，不存在则返回0
     */
    public int minSubArrayLen(int target, int[] nums) {
        // 滑动窗口的左指针
        int left = 0;
        // 滑动窗口的右指针
        int right = 0;

        // 记录满足条件的最短长度，初始化为最大整数，方便取最小值
        int res = Integer.MAX_VALUE;
        // 当前滑动窗口内所有元素的总和
        int sum = 0;

        // 遍历数组，右指针不断向右扩展
        while (right < nums.length) {
            // 将右指针指向的元素加入窗口和中，并移动右指针
            sum += nums[right++];

            // 当当前窗口的和满足条件（大于等于目标值）时
            // 尝试通过移动左指针来缩小窗口，寻找可能的最短长度
            while (sum >= target) {
                // 更新结果：取当前窗口长度 (right - left) 与历史最小长度的较小值
                // 注意：由于上面执行了 right++，当前的窗口范围实际上是 [left, right-1]，长度正是 right - left
                res = Math.min(res, right - left);

                // 准备收缩左边界：先从总和中减去左指针指向的值
                sum -= nums[left];
                // 左指针右移，缩小窗口
                left++;
            }
        }

        // 如果 res 还是初始值，说明没有找到满足条件的子数组，返回 0
        // 否则返回找到的最小长度 res
        return res == Integer.MAX_VALUE ? 0 : res;
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

    /**
     * 寻找数组中所有和为目标值的数对（从指定起始位置开始）
     * <p>
     * 算法思路：
     * 1. 使用双指针技巧（左右指针），适用于有序数组
     * 2. 左指针从指定起始位置start开始向右移动，右指针从数组末尾向左移动
     * 3. 根据当前两数之和与目标值的比较结果移动指针：
     * - 如果和等于目标值：找到一对解，记录结果并跳过重复元素避免重复解
     * - 如果和小于目标值：左指针右移增大和值
     * - 如果和大于目标值：右指针左移减小和值
     * 4. 为了避免重复结果，在找到一个有效数对后，跳过所有相同的元素
     * <p>
     * 时间复杂度：O(n)，其中n是数组长度（虽然有嵌套循环，但每个元素最多被访问一次）
     * 空间复杂度：O(1)，不考虑结果列表的空间
     *
     * @param nums   输入的有序整数数组
     * @param start  搜索的起始位置
     * @param target 目标和
     * @return 所有和为 target 的数对组成的列表
     */
    public List<List<Integer>> twoSumTarget(int[] nums, int start, int target) {
        // 创建结果列表，用于存储所有满足条件的数对
        List<List<Integer>> res = new ArrayList<>();

        // 获取数组长度，用于初始化右指针
        int n = nums.length;

        // 初始化双指针：左指针指向指定起始位置，右指针指向数组末尾
        int low = start, high = n - 1;

        // 当左指针小于右指针时继续循环（确保不会重复使用同一个元素）
        while (low < high) {
            // 计算当前左右指针所指元素的和
            int sum = nums[low] + nums[high];

            // 保存当前左右指针的值，用于后续去重判断
            int left = nums[low], right = nums[high];

            // 情况1：当前和等于目标值，找到一个有效的数对
            if (sum == target) {
                // 将找到的数对添加到结果列表中
                res.add(Arrays.asList(nums[low], nums[high]));

                // 跳过所有与当前左指针值相同的元素，避免重复结果
                // 继续移动左指针直到遇到不同的值或追上右指针
                while (low < high && nums[low] == left) {
                    low++;
                }

                // 跳过所有与当前右指针值相同的元素，避免重复结果
                // 继续移动右指针直到遇到不同的值或追上左指针
                while (low < high && nums[high] == right) {
                    high--;
                }
            }
            // 情况2：当前和小于目标值，需要增大和值
            else if (sum < target) {
                // 左指针右移，选择更大的数
                low++;
            }
            // 情况3：当前和大于目标值，需要减小和值
            else {
                // 右指针左移，选择更小的数
                high--;
            }
        }

        // 返回所有找到的数对
        return res;
    }

    /**
     * 寻找数组中所有和为0的三元组
     * <p>
     * 算法思路：
     * 1. 复用threeSumTarget方法，将目标和设为0
     * 2. 通过固定一个数，转化为两数之和的问题
     * <p>
     * 时间复杂度：O(n²)，其中n是数组长度
     * 空间复杂度：O(1)，不考虑结果列表的空间
     *
     * @param nums 输入的整数数组
     * @return 所有和为0的不重复三元组组成的列表
     */
    public List<List<Integer>> threeSum(int[] nums) {
        // 调用threeSumTarget方法，目标和设为0
        return threeSumTarget(nums, 0);
    }

    /**
     * 寻找数组中所有和为目标值的三元组
     * <p>
     * 算法思路：
     * 1. 排序：先对数组进行排序，便于使用双指针技巧和去重
     * 2. 固定一个数：遍历数组，将nums[i]作为三元组的第一个数
     * 3. 转化为两数之和：在nums[i+1:]中寻找和为target-nums[i]的两元组
     * 4. 合并结果：将nums[i]与找到的两元组合并成三元组
     * 5. 去重：跳过重复的nums[i]值，避免重复的三元组
     * <p>
     * 时间复杂度：O(n²)，其中n是数组长度（排序O(nlogn)，双重循环O(n²)）
     * 空间复杂度：O(1)，不考虑结果列表的空间
     *
     * @param nums   输入的整数数组
     * @param target 目标和
     * @return 所有和为 target 的不重复三元组组成的列表
     */
    public List<List<Integer>> threeSumTarget(int[] nums, int target) {
        // 对数组进行排序，这是使用双指针技巧的前提
        Arrays.sort(nums);

        // 创建结果列表，用于存储所有满足条件的三元组
        List<List<Integer>> res = new ArrayList<>();

        // 遍历数组，将每个元素作为三元组的第一个数
        for (int i = 0; i < nums.length; i++) {
            // 在nums[i+1:]范围内寻找和为target-nums[i]的两元组
            // 这里复用了twoSumTarget方法，将问题转化为两数之和
            List<List<Integer>> tuples = twoSumTarget(nums, i + 1, target - nums[i]);

            // 遍历找到的所有两元组，与当前nums[i]组成三元组
            for (List<Integer> tuple : tuples) {
                // 将nums[i]与两元组tuple中的两个元素组成新的三元组
                // tuple.get(0)和tuple.get(1)是twoSumTarget找到的两个数
                res.add(Arrays.asList(nums[i], tuple.get(0), tuple.get(1)));
            }

            // 跳过重复的nums[i]值，避免产生重复的三元组
            // 当nums[i]与nums[i+1]相等时，继续递增i直到找到不同的值
            while (i + 1 < nums.length && nums[i] == nums[i + 1]) {
                i++;
            }
        }

        // 返回所有找到的三元组
        return res;
    }

    /**
     * 寻找数组中所有和为目标值的四元组
     * <p>
     * 算法思路：
     * 1. 排序：先对数组进行排序，便于使用双指针技巧和去重
     * 2. 固定一个数：遍历数组，将nums[i]作为四元组的第一个数
     * 3. 转化为三数之和：在nums[i+1:]中寻找和为target-nums[i]的三元组
     * 4. 合并结果：将nums[i]与找到的三元组合并成四元组
     * 5. 去重：跳过重复的nums[i]值，避免重复的四元组
     * <p>
     * 时间复杂度：O(n³)，其中n是数组长度（排序O(nlogn)，三重循环O(n³)）
     * 空间复杂度：O(1)，不考虑结果列表的空间
     *
     * @param nums   输入的整数数组
     * @param target 目标和
     * @return 所有和为 target 的不重复四元组组成的列表
     */
    public List<List<Integer>> fourSum(int[] nums, int target) {
        // 初始化结果列表，用于存储所有满足条件的四元组
        List<List<Integer>> res = new ArrayList<>();

        // 首先对数组进行排序，这是使用双指针算法和去重策略的基础
        // 排序后可以利用数组的有序性进行高效的搜索和去重
        Arrays.sort(nums);

        // 遍历数组，将每个元素作为四元组的第一个元素
        // 通过固定第一个元素，将四数之和问题转化为三数之和问题
        for (int i = 0; i < nums.length; i++) {
            // 调用threeSumTarget方法，在nums[i+1:]子数组中寻找和为(target-nums[i])的三元组
            // 这样nums[i] + 三元组的和 = target，构成一个有效的四元组
            List<List<Integer>> triples = threeSumTarget(nums, target - nums[i]);

            // 遍历所有找到的三元组，将当前固定的nums[i]与每个三元组合并成四元组
            for (List<Integer> triple : triples) {
                // 创建新的四元组列表，将当前元素nums[i]添加到三元组前面
                // 这里创建新列表避免修改原始的triple，防止对后续操作造成影响
                List<Integer> quadruplet = new ArrayList<>();
                quadruplet.add(nums[i]);
                quadruplet.addAll(triple);

                // 将完整的四元组添加到结果列表中
                res.add(quadruplet);
            }

            // 跳过重复的nums[i]值，避免产生重复的四元组
            // 当当前元素与下一个元素相等时，继续递增i直到找到不同的值
            // 这是去重的关键步骤，确保不会出现重复的四元组
            while (i + 1 < nums.length && nums[i] == nums[i + 1]) {
                i++;
            }
        }

        // 返回所有找到的不重复四元组
        return res;
    }

    /**
     * 接雨水问题 - 暴力解法
     * <p>
     * 算法思路：
     * 1. 对于每个位置i，计算其能够存储的雨水量
     * 2. 每个位置能存储的雨水量 = min(左侧最高柱子, 右侧最高柱子) - 当前位置高度
     * 3. 遍历每个位置，分别向左向右寻找最大值，计算存储的雨水量
     * <p>
     * 为什么i从1开始遍历到n-2：
     * - i从1开始：因为索引0的位置左边没有柱子，无法形成凹槽，所以无法存储雨水
     * - i到n-2结束：因为索引n-1的位置右边没有柱子，无法形成凹槽，所以无法存储雨水
     * - 只有中间的位置才有可能被左右两侧的柱子包围形成储水区域
     * <p>
     * 时间复杂度：O(n²)，对于每个位置都要遍历整个数组寻找左右最大值
     * 空间复杂度：O(1)，只使用了常数级别的额外空间
     *
     * @param height 表示柱子高度的数组
     * @return 能够接住的雨水总量
     */
    public int trap(int[] height) {
        int n = height.length; // 获取数组长度
        int res = 0;           // 记录总的雨水量

        // 从索引1开始遍历到n-2，因为首尾两个位置无法存储雨水
        for (int i = 1; i < n - 1; i++) {
            int l_max = 0; // 记录位置 i 左侧的最大高度
            int r_max = 0; // 记录位置 i 右侧的最大高度

            // 从位置i开始向右遍历到数组末尾，寻找右侧最大高度
            for (int j = i; j < n; j++) {
                r_max = Math.max(r_max, height[j]); // 更新右侧最大高度
            }

            // 从位置i开始向左遍历到数组开头，寻找左侧最大高度
            for (int j = i; j >= 0; j--) {
                l_max = Math.max(l_max, height[j]); // 更新左侧最大高度
            }

            // 位置i能存储的雨水量 = min(左侧最大高度, 右侧最大高度) - 当前位置高度
            // 这是因为水的高度取决于较矮的那一侧（木桶效应）
            res += Math.min(r_max, l_max) - height[i];
        }
        return res; // 返回总的雨水量
    }

    /**
     * 接雨水问题 - 动态规划优化解法
     * <p>
     * 算法思路：
     * 1. 预处理：使用两个数组分别记录每个位置左侧和右侧的最大高度
     * 2. 对于每个位置i，其能够存储的雨水量 = min(左侧最大高度, 右侧最大高度) - 当前位置高度
     * 3. 遍历每个位置，累加可存储的雨水量
     * <p>
     * 优化点：
     * - 暴力解法中每次都要向左右遍历寻找最大值，时间复杂度O(n²)
     * - 本方法预先计算并存储每个位置的左右最大值，查询时间为O(1)，总体时间复杂度降为O(n)
     * <p>
     * 时间复杂度：O(n)，其中n是数组长度（三次遍历：计算左最大值、计算右最大值、计算雨水量）
     * 空间复杂度：O(n)，需要两个额外数组存储左右最大值
     *
     * @param height 表示柱子高度的数组
     * @return 能够接住的雨水总量
     */
    public int trap2(int[] height) {
        int n = height.length; // 获取数组长度
        int res = 0;           // 记录总的雨水量

        // 创建两个数组分别记录每个位置左侧和右侧的最大高度
        int[] l_max = new int[n];  // l_max[i] 表示位置 i 及其左侧的最大高度
        int[] r_max = new int[n];  // r_max[i] 表示位置 i 及其右侧的最大高度

        // 初始化边界值
        l_max[0] = height[0];         // 位置0左侧的最大高度就是它自己
        r_max[n - 1] = height[n - 1]; // 位置n-1右侧的最大高度就是它自己

        // 从左到右遍历，计算每个位置及其左侧的最大高度
        for (int i = 1; i < n; i++) {
            // 位置i的左侧最大高度 = max(位置i-1的左侧最大高度, 位置i的高度)
            // 这样可以保证l_max[i]包含了从0到i的所有柱子的最大值
            l_max[i] = Math.max(l_max[i - 1], height[i]);
        }

        // 从右到左遍历，计算每个位置及其右侧的最大高度
        for (int i = n - 2; i >= 0; i--) {
            // 位置i的右侧最大高度 = max(位置i+1的右侧最大高度, 位置i的高度)
            // 这样可以保证r_max[i]包含了从i到n-1的所有柱子的最大值
            r_max[i] = Math.max(r_max[i + 1], height[i]);
        }

        // 遍历每个位置（除了首尾两个位置，因为它们无法形成凹槽），计算雨水量
        for (int i = 1; i < n - 1; i++) {
            // 位置i能存储的雨水量 = min(左侧最大高度, 右侧最大高度) - 当前位置高度
            // 这是因为水的高度取决于较矮的那一侧（木桶效应）
            // 只有当前高度小于左右两侧较低的那个高度时，才能存储雨水
            res += Math.min(l_max[i], r_max[i]) - height[i];
        }

        return res; // 返回总的雨水量
    }

    /**
     * 接雨水问题 - 双指针优化解法
     * <p>
     * 算法思路：
     * 1. 使用双指针从两端向中间移动，同时维护左右两侧的最大高度
     * 2. 对于每个位置，其能存储的雨水量取决于较矮的一侧（木桶效应）
     * 3. 总是处理当前较矮的一侧，因为该侧的储水量已经确定（由当前较矮侧决定）
     * 4. 移动指针继续处理下一个位置
     * <p>
     * 优化点：
     * - 不需要预计算左右最大值数组，节省空间复杂度至O(1)
     * - 一次遍历完成，时间复杂度O(n)
     * - 通过双指针避免重复计算，提高效率
     * <p>
     * 时间复杂度：O(n)，其中n是数组长度
     * 空间复杂度：O(1)，只使用了常数级别的额外空间
     *
     * @param height 表示柱子高度的数组
     * @return 能够接住的雨水总量
     */
    public int trap3(int[] height) {
        int n = height.length; // 获取数组长度
        int res = 0;           // 记录总的雨水量

        int right = n - 1;     // 右指针从数组末尾开始
        int left = 0;          // 左指针从数组开头开始

        int l_max = 0;         // 记录左指针左侧的最大高度（包含左指针当前位置）
        int r_max = 0;         // 记录右指针右侧的最大高度（包含右指针当前位置）

        // 双指针相遇前持续循环
        // 当left == right时，说明所有位置都已处理完毕
        while (left < right) {
            // 更新左指针左侧的最大高度
            // 包括当前位置height[left]在内的左侧最大值
            l_max = Math.max(l_max, height[left]);

            // 更新右指针右侧的最大高度  
            // 包括当前位置height[right]在内的右侧最大值
            r_max = Math.max(r_max, height[right]);

            // 关键逻辑：基于木桶效应原理
            // 水位高度取决于较矮的一侧，所以我们优先处理较矮的一侧
            if (l_max < r_max) {
                // 左侧最大值较小，说明left位置的储水量仅由左侧决定
                // 因为右侧必定存在更高的柱子（r_max > l_max），所以不用担心右侧"漏水"
                // 当前位置能存储的雨水 = 左侧最大高度 - 当前位置高度
                res += l_max - height[left];

                // 处理完left位置后，左指针右移
                left++;
            } else {
                // 右侧最大值较小或相等，说明right位置的储水量仅由右侧决定
                // 因为左侧必定存在不低于当前r_max的柱子（l_max >= r_max）
                // 当前位置能存储的雨水 = 右侧最大高度 - 当前位置高度
                res += r_max - height[right];

                // 处理完right位置后，右指针左移
                right--;
            }
        }
        return res; // 返回总的雨水量
    }

    /**
     * 寻找能盛最多水的容器
     * <p>
     * 算法思路：
     * 1. 使用双指针技巧（左右指针），初始时左指针指向数组开头，右指针指向数组末尾
     * 2. 计算当前两个指针位置形成的容器面积：min(高度[left], 高度[right]) * (right - left)
     * 3. 更新最大面积
     * 4. 移动较短一边的指针：因为容器的容量受限于较短的一边，移动较长边只会减少宽度而不一定能增加容量
     * 5. 重复步骤2-4直到两个指针相遇
     * <p>
     * 核心思想：贪心策略 - 总是移动较短的一边，因为移动较短边可能找到更高的高度从而增加容量
     * 而移动较长边不可能增加容量，因为容量仍然受限于较短边的高度。
     * <p>
     * 时间复杂度：O(n)，其中 n 是数组的长度，每个元素最多被访问一次
     * 空间复杂度：O(1)，只使用了常数个额外变量
     *
     * @param height 表示垂直线条高度的数组
     * @return 容器能够储存的最大水量
     */
    public int maxArea(int[] height) {
        int n = height.length;         // 获取数组长度
        int res = 0;                   // 记录最大面积，初始为0
        int left = 0, right = n - 1;   // 初始化双指针：左指针指向开头，右指针指向末尾

        // 当左指针小于右指针时继续循环（确保两个指针没有相遇）
        while (left < right) {
            // 计算当前两个指针位置形成的容器面积

            // 容器高度 = 两个高度中的较小值（短板效应）
            // 容器宽度 = 两个指针之间的距离
            // 面积 = 高度 × 宽度
            int cur_area = Math.min(height[left], height[right]) * (right - left);

            // 更新最大面积：取当前面积和历史最大面积中的较大值
            res = Math.max(cur_area, res);

            // 关键决策：移动较短一边的指针
            // 如果左边高度小于右边高度，移动左指针
            if (height[left] < height[right]) {
                left++;  // 左指针右移，寻找可能更高的高度
            } else {
                // 否则移动右指针（右边高度小于等于左边高度）
                right--; // 右指针左移，寻找可能更高的高度
            }
            // 为什么要移动较短边？
            // 因为容器容量 = min(h1, h2) * width
            // 如果移动较高边，新的容量 = min(h1, new_h2) * (width-1)
            // 由于new_width < old_width，且min(h1, new_h2) <= min(h1, h2)，所以新容量一定小于原容量
            // 只有移动较短边，才有可能找到更高的高度从而增加容量
        }

        return res; // 返回找到的最大面积
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