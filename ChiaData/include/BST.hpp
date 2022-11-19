#include "List.hpp"
#include "Types/Types.hpp"

namespace ChiaData
{
template <class Key, class Value> class BST
{
  protected:
    struct Node
    {
        BST<Key, Value> *pTree;
        Node *pLeft;
        Node *pRight;
        Key key;
        Value value;

        template <class ValueType>
        Node(BST<Key, Value> &tree, const Key &key, ValueType &&value, Node *pLeft = nullptr, Node *pRight = nullptr)
            : pTree(&tree), pLeft(pLeft), pRight(pRight), key(key), value(Types::Forward<Value>(value))
        {
        }
    };

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

    Node *FindMinNode(Node *pSubtree, Node *pParent, Node **ppParent)
    {
        if (pSubtree->pLeft)
            return FindMinNode(pSubtree->pLeft, pSubtree, ppParent);
        if (ppParent)
            *ppParent = pParent;
        return pSubtree;
    }

    Node *FindSuccessor(Node *pNode, Node **ppParent = nullptr)
    {
        if (ppParent)
            *ppParent = nullptr;
        if (!pNode || !pNode->pRight)
            return nullptr;
        return FindMinNode(pNode->pRight, pNode, ppParent);
    }

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

    static bool IsLeaf(Node *pNode)
    {
        return pNode && !pNode->pLeft && !pNode->pRight;
    }

    Node *pRoot;

    size_t nNodes;

  public:
    BST() : pRoot(nullptr), nNodes(0)
    {
    }

    virtual ~BST()
    {
        DeleteTree(pRoot);
    }

    bool IsEmpty() const
    {
        return pRoot == nullptr;
    }

    bool Contains(const Key &key) const
    {
        return FindNodePtr(key) != nullptr;
    }

    Value &Get(const Key &key)
    {
        return FindNodePtr(key)->value;
    }

    const Value &Get(const Key &key) const
    {
        return FindNodePtr(key)->value;
    }

    Value &operator[](const Key &key)
    {
        return Get(key);
    }

    const Value &operator[](const Key &key) const
    {
        return Get(key);
    }

    template <class ValueType> void Insert(const Key &key, ValueType &&value)
    {
        if (!pRoot)
        {
            pRoot = new Node(*this, key, Types::Forward<Value>(value));
            nNodes++;
            return;
        }
        Node *pNodeFound, *pParent;
        pNodeFound = FindNodePtr(key, &pParent);
        if (!pNodeFound)
        {
            if (key > pParent->key)
                pParent->pRight = new Node(*this, key, Types::Forward<Value>(value));
            else
                pParent->pLeft = new Node(*this, key, Types::Forward<Value>(value));
            nNodes++;
            return;
        }
        pNodeFound->value = Types::Forward<Value>(value);
    }

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
