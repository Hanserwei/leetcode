package linked_list;

public class MyLinkedList<E> {
    // 虚拟头节点
    private Node<E> head;
    private Node<E> tail;
    private int size;

    public MyLinkedList() {
        this.head = new Node<>(null);
        this.tail = new Node<>(null);
        head.next = tail;
        tail.prev = head;
        this.size = 0;
        /*
         * 初始化双向链表，两个虚节点，head和tail
         * dummy(head) <--> dummy(tail)
         * */
    }

    // 尾增
    public void addLast(E e) {
        // 新节点
        Node<E> x = new Node<>(e);
        // 拿到最后一个节点
        Node<E> prevNode = tail.prev;
        prevNode.next = x;
        x.prev = prevNode;

        // 虚尾节拼接
        x.next = tail;
        tail.prev = x;
        // 长度+1
        size++;
    }

    // 头增
    public void addFirst(E e) {
        // 新节点
        Node<E> x = new Node<>(e);
        // 拿到第一个节点
        Node<E> next = head.next;
        next.prev = x;
        x.next = next;

        // 拼接头虚节点
        head.next = x;
        x.prev = head;

        // 长度+1
        size++;
    }

    // 中间增
    public void add(int index, E e) {
        // 判断索引是否有效
        checkPositionIndex(index);
        // 新节点
        Node<E> x = new Node<>(e);
        if (index == size) {
            addLast(e);
            return;
        }
        // 找到 index 对应的节点
        Node<E> p = getNode(index);
        Node<E> temp = p.prev;
        // 放入新节点
        p.prev = x;
        temp.next = x;

        x.prev = temp;
        x.next = p;
        size++;
    }

    // 删除头节点
    public void removeFirst() {
        if (head.next == tail) {
            throw new RuntimeException("No such element");
        }
        Node<E> next = head.next;

        head.next = next.next;
        next.next.prev = head;

        next.prev = null;
        next.next = null;
        size--;
    }

    // 删除尾节点
    public void removeLast() {
        if (tail.prev == head) {
            throw new RuntimeException("No such element");
        }
        Node<E> prev = tail.prev;
        prev.prev.next = prev.next;
        prev.next.prev = prev.prev;

        prev.prev = null;
        prev.next = null;
        size--;
    }

    // 删除中间节点
    public void removeAt(int index) {
        checkElementIndex(index);
        Node<E> x = getNode(index);
        x.prev.next = x.next;
        x.next.prev = x.prev;

        x.prev = null;
        x.next = null;

        size--;
    }

    // 改
    public void replaceAt(int index, E e) {
        checkElementIndex(index);
        Node<E> x = getNode(index);

        Node<E> node = new Node<>(e);
        x.prev.next = node;
        x.next.prev = node;

        node.prev = x.prev;
        node.next = x.next;

        x.prev = null;
        x.next = null;
    }
    
    // 查
    public E get(int index) {
        checkElementIndex(index);
        Node<E> x = getNode(index);
        return x.data;
    }

    private Node<E> getNode(int index) {
        checkElementIndex(index);
        Node<E> cur = head.next;
        for (int i = 0; i < index; i++) {
            cur = cur.next;
        }
        return cur;
    }

    private void checkElementIndex(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size);
        }
    }

    private void checkPositionIndex(int index) {
        if (index < 0 || index > size) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size);
        }
    }

    //内部类
    static class Node<E> {
        E data;
        Node<E> next;
        Node<E> prev;

        public Node(E data) {
            this.data = data;
        }
    }

    public static void main(String[] args) {
        System.out.println("测试自定义双向链表 MyLinkedList");
        
        // 创建链表实例
        MyLinkedList<String> list = new MyLinkedList<>();
        
        // 测试 addLast
        System.out.println("\n1. 测试 addLast:");
        list.addLast("A");
        list.addLast("B");
        list.addLast("C");
        System.out.println("添加 A, B, C 后，链表大小: " + list.size);
        printList(list);
        
        // 测试 addFirst
        System.out.println("\n2. 测试 addFirst:");
        list.addFirst("First");
        System.out.println("头部添加 'First' 后:");
        printList(list);
        
        // 测试 add(index, e)
        System.out.println("\n3. 测试 add(index, e):");
        list.add(2, "AtIndex2");
        System.out.println("在索引2处添加 'AtIndex2' 后:");
        printList(list);
        
        // 测试 get
        System.out.println("\n4. 测试 get:");
        for (int i = 0; i < list.size; i++) {
            System.out.println("索引 " + i + " 的元素: " + list.get(i));
        }
        
        // 测试 replaceAt
        System.out.println("\n5. 测试 replaceAt:");
        list.replaceAt(1, "ReplacedB");
        System.out.println("替换索引1的元素为 'ReplacedB' 后:");
        printList(list);
        
        // 测试 removeFirst
        System.out.println("\n6. 测试 removeFirst:");
        list.removeFirst();
        System.out.println("删除第一个元素后:");
        printList(list);
        
        // 测试 removeLast
        System.out.println("\n7. 测试 removeLast:");
        list.removeLast();
        System.out.println("删除最后一个元素后:");
        printList(list);
        
        // 测试 removeAt
        System.out.println("\n8. 测试 removeAt:");
        list.removeAt(1);
        System.out.println("删除索引1的元素后:");
        printList(list);
        
        // 边界测试
        System.out.println("\n9. 边界测试:");
        MyLinkedList<Integer> numList = new MyLinkedList<>();
        numList.addLast(1);
        numList.addLast(2);
        numList.addLast(3);
        System.out.println("数字链表: ");
        printList(numList);
        
        // 测试异常情况
        System.out.println("\n10. 异常情况测试:");
        try {
            numList.get(10); // 索引超出范围
        } catch (Exception e) {
            System.out.println("捕获异常: " + e.getMessage());
        }
        
        try {
            numList.add(-1, 0); // 负索引
        } catch (Exception e) {
            System.out.println("捕获异常: " + e.getMessage());
        }
        
        // 完整操作序列测试
        System.out.println("\n11. 完整操作序列测试:");
        MyLinkedList<Character> charList = new MyLinkedList<>();
        charList.addLast('a');
        charList.addLast('b');
        charList.addFirst('x');
        charList.add(1, 'y');
        System.out.println("执行 addLast('a'), addLast('b'), addFirst('x'), add(1,'y') 后:");
        printList(charList);
        
        charList.removeAt(2);
        charList.replaceAt(0, 'z');
        System.out.println("执行 removeAt(2), replaceAt(0,'z') 后:");
        printList(charList);
    }
    
    // 辅助方法：打印链表内容
    private static <T> void printList(MyLinkedList<T> list) {
        System.out.print("链表内容 [");
        for (int i = 0; i < list.size; i++) {
            System.out.print(list.get(i));
            if (i < list.size - 1) {
                System.out.print(", ");
            }
        }
        System.out.println("] (size: " + list.size + ")");
    }
}
