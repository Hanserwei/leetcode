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
}

class NumArray {

    private final int[] preSum;

    public NumArray(int[] nums) {
        preSum = new int[nums.length + 1];

        for (int i = 1; i < nums.length; i++) {
            preSum[i] = preSum[i - 1] + nums[i - 1];
        }
    }

    // 查询闭区间 [left, right] 的累加和
    public int sumRange(int left, int right) {
        return preSum[right + 1] - preSum[left];
    }
}

class ListNode {
    int val;
    ListNode next;

    ListNode(int val) {
        this.val = val;
    }

}