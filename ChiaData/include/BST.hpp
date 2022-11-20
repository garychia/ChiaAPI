#include "List.hpp"
#include "Types/Types.hpp"

namespace ChiaData
{
/**
 * @brief Binary Search Tree
 *
 * @tparam Key the type of keys used to access a node.
 * @tparam Value the type of values stored in the nodes.
 */
template <class Key, class Value> class BST
{
  protected:
    /**
     * @brief Presents a node in the binary search tree.
     */
    struct Node
    {
        /**
         * @brief the left child
         */
        Node *pLeft;
        /**
         * @brief the right child
         */
        Node *pRight;
        /**
         * @brief the key that identifies this node.
         */
        Key key;
        /**
         * @brief the value this node stores.
         */
        Value value;

        /**
         * @brief Construct a new Node object
         *
         * @tparam ValueType the type of data stored in the node.
         * @param key the key used to identifies the node.
         * @param value the data to be stored in the node.
         * @param pLeft the left child.
         * @param pRight the right child.
         */
        template <class ValueType>
        Node(const Key &key, ValueType &&value, Node *pLeft = nullptr, Node *pRight = nullptr)
            : pLeft(pLeft), pRight(pRight), key(key), value(Types::Forward<Value>(value))
        {
        }
    };

    /**
     * @brief Delete a subtree in the binary search tree
     *
     * @param pRootNode the root of the subtree.
     */
    virtual void DeleteTree(Node *pRootNode)
    {
        List<Node *> pNodesToDelete;
        if (pRootNode)
            pNodesToDelete.Append(pRootNode);
        if (pRoot == pRootNode)
            pRoot = nullptr;
        while (!pNodesToDelete.IsEmpty())
        {
            Node *pNode = *pNodesToDelete.First();
            pNodesToDelete.RemoveFirst();
            if (pNode->pLeft)
                pNodesToDelete.Append(pNode->pLeft);
            if (pNode->pRight)
                pNodesToDelete.Append(pNode->pRight);
            delete pNode;
            nNodes--;
        }
    }

    /**
     * @brief Find the pointer of node associated with the given key.
     *
     * @param key the key used to find the assocated node.
     * @param ppParent (OPTIONAL) the pointer of pointer of the parent of resulting node.
     * @return Node* the pointer of node associated with the key if found. Otherwise, NULL is returned.
     */
    Node *FindNodePtr(const Key &key, Node **ppParent = nullptr)
    {
        if (ppParent)
            *ppParent = nullptr;
        auto pCurrent = pRoot;
        while (pCurrent)
        {
            if (pCurrent->key == key)
            {
                return pCurrent;
            }
            else if (key > pCurrent->key)
            {
                if (ppParent)
                    *ppParent = pCurrent;
                pCurrent = pCurrent->pRight;
            }
            else
            {
                if (ppParent)
                    *ppParent = pCurrent;
                pCurrent = pCurrent->pLeft;
            }
        }
        return pCurrent;
    }

    /**
     * @brief Find the node with the minimum key in a subtree.
     *
     * @param pSubtree the root node of subtree.
     * @param pParent the parent node of root.
     * @param ppParent (OPTIONAL) the pointer of pointer of the parent node.
     * @return Node* the node with the minimum key in the subtree.
     */
    static Node *FindMinNode(Node *pSubtree, Node *pParent, Node **ppParent = nullptr)
    {
        if (pSubtree->pLeft)
            return FindMinNode(pSubtree->pLeft, pSubtree, ppParent);
        if (ppParent)
            *ppParent = pParent;
        return pSubtree;
    }

    /**
     * @brief Find the successor of a node.
     *
     * @param pNode the node.
     * @param ppParent (OPTIONAL) the pointer of pointer of the parent node.
     * @return Node* the successor node.
     */
    static Node *FindSuccessor(Node *pNode, Node **ppParent = nullptr)
    {
        if (ppParent)
            *ppParent = nullptr;
        if (!pNode || !pNode->pRight)
            return nullptr;
        return FindMinNode(pNode->pRight, pNode, ppParent);
    }

    /**
     * @brief Delete a node
     *
     * @param pNode the node.
     * @return Node* the node that replaced the deleted node.
     */
    Node *DeleteNode(Node *pNode)
    {
        nNodes--;
        if (IsLeaf(pNode))
        {
            delete pNode;
            return nullptr;
        }
        else if (pNode->pLeft && pNode->pLeft)
        {
            Node *pSuccessorParent;
            Node *pSuccessor = FindSuccessor(pNode, &pSuccessorParent);
            Node *pSuccessorRight = pSuccessor->pRight;
            pSuccessorParent->pLeft = pSuccessorRight;
            pSuccessor->pLeft = pNode->pLeft;
            pSuccessor->pRight = pNode->pRight;
            delete pNode;
            return pSuccessor;
        }
        else if (pNode->pLeft)
        {
            Node *pLeftNode = pNode->pLeft;
            delete pNode;
            return pLeftNode;
        }
        else
        {
            Node *pRightNode = pNode->pRight;
            delete pNode;
            return pRightNode;
        }
    }

    /**
     * @brief Check if a node is a leaf node.
     *
     * @param pNode the node.
     * @return true if the node is a leaf node.
     * @return false otherwise.
     */
    static bool IsLeaf(Node *pNode)
    {
        return pNode && !pNode->pLeft && !pNode->pRight;
    }

    /**
     * @brief root node
     */
    Node *pRoot;

    /**
     * @brief number of nodes stored in the tree
     *
     */
    size_t nNodes;

  public:
    /**
     * @brief Construct a new BST object
     */
    BST() : pRoot(nullptr), nNodes(0)
    {
    }

    /**
     * @brief Destroy the BST object
     */
    virtual ~BST()
    {
        DeleteTree(pRoot);
    }

    /**
     * @brief Check if the BST is empty
     *
     * @return true if it is empty.
     * @return false otherwise.
     */
    bool IsEmpty() const
    {
        return pRoot == nullptr;
    }

    /**
     * @brief Check if there is a node associated with the given key
     *
     * @param key the key.
     * @return true if there is a node associated with the key.
     * @return false otherwise.
     */
    bool Contains(const Key &key) const
    {
        return FindNodePtr(key) != nullptr;
    }

    /**
     * @brief Get the value from the node associated with the given key
     *
     * @param key the key.
     * @return Value& the value.
     */
    Value &Get(const Key &key)
    {
        return FindNodePtr(key)->value;
    }

    /**
     * @brief Get the value from the node associated with the given key
     *
     * @param key the key.
     * @return const Value& the value.
     */
    const Value &Get(const Key &key) const
    {
        return FindNodePtr(key)->value;
    }

    /**
     * @brief Get the value from the node associated with the given key
     *
     * @param key the key.
     * @return Value& the value.
     */
    Value &operator[](const Key &key)
    {
        return Get(key);
    }

    /**
     * @brief Get the value from the node associated with the given key
     *
     * @param key the key.
     * @return const Value& the value.
     */
    const Value &operator[](const Key &key) const
    {
        return Get(key);
    }

    /**
     * @brief Insert a key-value pair into the BST
     *
     * @tparam ValueType the type of value.
     * @param key the key.
     * @param value the value.
     */
    template <class ValueType> void Insert(const Key &key, ValueType &&value)
    {
        if (!pRoot)
        {
            pRoot = new Node(key, Types::Forward<Value>(value));
            nNodes++;
            return;
        }
        Node *pNodeFound, *pParent;
        pNodeFound = FindNodePtr(key, &pParent);
        if (!pNodeFound)
        {
            if (key > pParent->key)
                pParent->pRight = new Node(key, Types::Forward<Value>(value));
            else
                pParent->pLeft = new Node(key, Types::Forward<Value>(value));
            nNodes++;
            return;
        }
        pNodeFound->value = Types::Forward<Value>(value);
    }

    /**
     * @brief Delete the node associated with the given key
     *
     * @param key the key.
     */
    void Delete(const Key &key)
    {
        Node *pParent;
        Node *pNode = FindNodePtr(key, &pParent);
        if (!pNode)
            return;
        const bool isLeftOfParent = pNode == pParent->pLeft;
        const bool isRoot = pNode == pRoot;
        Node *pNewSubtree = DeleteNode(pNode);
        if (isRoot)
            pRoot = pNewSubtree;
        if (isLeftOfParent)
            pParent->pLeft = pNewSubtree;
        else
            pParent->pRight = pNewSubtree;
    }
};

} // namespace ChiaData
