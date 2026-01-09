package linked_list.leetcode;

import java.util.*;

class Solution {
    /**
     * 找到两个有序数组中和最小的k个数对
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
     * @return 和最小的k个数对
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
        // NumArray内部维护一个前缀和数组preSum，其中preSum[i]表示nums[0]到nums[i-1]的累加和
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
}

/**
 * 使用前缀和数组实现的数组区间求和类
 * <p>
 * 算法思路：
 * 1. 预处理：构建前缀和数组preSum，其中preSum[i]表示原数组nums[0]到nums[i-1]的累加和
 * 2. 查询：利用前缀和的性质，区间[left, right]的和 = preSum[right+1] - preSum[left]
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
        for (int i = 1; i <= nums.length; i++) {
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
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                // 二维前缀和的计算公式：
                // preSum[i][j] = 左边区域的和 + 上面区域的和 - 重叠区域的和 + 当前位置的值
                // 即：preSum[i][j] = preSum[i-1][j] + preSum[i][j-1] - preSum[i-1][j-1] + matrix[i-1][j-1]
                preSum[i][j] = preSum[i - 1][j] + preSum[i][j - 1] - preSum[i - 1][j - 1] + matrix[i - 1][j - 1];
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