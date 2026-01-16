package linked_list;

import java.util.NoSuchElementException;

/**
 * ArrayQueue 类：使用数组实现一个支持动态扩容的环形队列。
 * 
 * @param <E> 队列中存储的元素类型
 */
public class ArrayQueue<E> {
    // 队列中当前元素的个数
    private int size;
    // 内部数组，用于存储队列元素
    private E[] data;
    // 默认的初始容量
    private final static int INIT_CAP = 2;

    // first 指向队列头部的元素位置
    // last 指向队列尾部下一个可插入的位置
    private int first, last;

    /**
     * 构造函数：初始化指定容量的队列
     * @param initCap 初始容量
     */
    public ArrayQueue(int initCap) {
        size = 0;
        data = (E[]) new Object[initCap];
        first = last = 0;
    }

    /**
     * 默认构造函数：使用默认容量初始化队列
     */
    public ArrayQueue() {
        this(INIT_CAP);
    }

    /**
     * 入队操作：在队列尾部添加一个元素
     * 如果数组已满，则自动扩容为当前容量的 2 倍
     * @param e 要添加的元素
     */
    public void enqueue(E e) {
        // 检查是否需要扩容
        if (size == data.length) {
            resize(size * 2);
        }

        // 在 last 指向的位置插入元素
        data[last] = e;
        // last 指针后移，如果超出数组边界则回到起始位置（环形结构）
        last++;
        if (last == data.length) {
            last = 0;
        }

        size++;
    }

    /**
     * 出队操作：移除并返回队列头部的元素
     * 如果队列为空，抛出异常。如果元素个数较少（为容量的 1/4），则进行缩容。
     * @return 被移除的头部元素
     * @throws NoSuchElementException 如果队列为空
     */
    public E dequeue() {
        if (isEmpty()) {
            throw new NoSuchElementException();
        }

        // 缩容逻辑：当元素个数少于容量的 1/4 时，缩容为原来的一半
        // 同时保证缩容后的容量不小于初始容量（可选优化）
        if (size == data.length / 4 && data.length / 2 > 0) {
            resize(data.length / 2);
        }

        // 获取头部元素，并将原位置置空以便垃圾回收
        E oldVal = data[first];
        data[first] = null;
        // first 指针后移，支持环形结构
        first++;
        if (first == data.length) {
            first = 0;
        }

        size--;
        return oldVal;
    }

    /**
     * 重新调整内部数组的大小
     * 将旧数组中的元素按照队列顺序拷贝到新数组中
     * @param newCap 新的容量大小
     */
    private void resize(int newCap) {
        E[] temp = (E[]) new Object[newCap];

        // 将原循环数组中的元素平铺拷贝到新数组
        // 索引 (first + i) % data.length 处理了环形绕回的情况
        for (int i = 0; i < size; i++) {
            temp[i] = data[(first + i) % data.length];
        }

        // 重置指针和数组引用
        first = 0;
        last = size;
        data = temp;
    }

    /**
     * 查看队列头部的元素（不移除）
     * @return 队列头部的元素
     * @throws NoSuchElementException 如果队列为空
     */
    public E peekFirst() {
        if (isEmpty()) {
            throw new NoSuchElementException();
        }
        return data[first];
    }

    /**
     * 查看队列尾部的最后一个元素（不移除）
     * @return 队列尾部的元素
     * @throws NoSuchElementException 如果队列为空
     */
    public E peekLast() {
        if (isEmpty()) {
            throw new NoSuchElementException();
        }
        // 由于 last 指向的是下一个插入位置，所以最后一个元素在 last - 1
        // 需要处理 last 为 0 时的边界情况
        if (last == 0) return data[data.length - 1];
        return data[last - 1];
    }

    /**
     * 返回队列中的元素个数
     */
    public int size() {
        return size;
    }

    /**
     * 判断队列是否为空
     */
    public boolean isEmpty() {
        return size == 0;
    }

}

/**
 * MyCircularQueue 类：基于 ArrayQueue 实现的定长循环队列。
 * 符合 LeetCode 循环队列题目要求。
 */
class MyCircularQueue {

    private ArrayQueue<Integer> q;
    private int maxCap;

    /**
     * 初始化队列，设置最大容量为 k
     */
    public MyCircularQueue(int k) {
        q = new ArrayQueue<>(k);
        maxCap = k;
    }

    /**
     * 向循环队列插入一个元素。如果成功插入则返回真。
     */
    public boolean enQueue(int value) {
        if (q.size() == maxCap) {
            return false;
        }
        q.enqueue(value);
        return true;
    }

    /**
     * 从循环队列中删除一个元素。如果成功删除则返回真。
     */
    public boolean deQueue() {
        if (q.isEmpty()) {
            return false;
        }
        q.dequeue();
        return true;
    }

    /**
     * 从队首获取元素。如果队列为空，返回 -1。
     */
    public int Front() {
        if (q.isEmpty()) {
            return -1;
        }
        return q.peekFirst();
    }

    /**
     * 获取队尾元素。如果队列为空，返回 -1。
     */
    public int Rear() {
        if (q.isEmpty()) {
            return -1;
        }
        return q.peekLast();
    }

    /**
     * 检查循环队列是否为空。
     */
    public boolean isEmpty() {
        return q.isEmpty();
    }

    /**
     * 检查循环队列是否已满。
     */
    public boolean isFull() {
        return q.size() == maxCap;
    }
}

