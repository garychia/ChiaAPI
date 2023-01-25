#ifndef HASH_TABLE_HPP
#define HASH_TABLE_HPP

#include <cstddef>

#include "Pair.hpp"
#include "Types/Types.hpp"

namespace ChiaData
{
/**
 * @brief A data structure that stores key-value pairs.
 *
 * @tparam Key the type of the key.
 * @tparam Value the type of the value.
 */
template <class Key, class Value> class HashTable
{
  private:
    /**
     * @brief A class that generates a hash value for a given key.
     */
    class KeyHash
    {
      public:
        /**
         * @brief Generate a hash value for a key.
         *
         * @param key the key.
         * @return std::size_t the hash value.
         */
        static std::size_t Generate(const Key &key)
        {
            return ((std::size_t)key & 0xff) * 17 + (((std::size_t)key >> 8) & 0xff) * 13 +
                   (((std::size_t)key >> 16) & 0xff) * 3;
        }
    };

    /**
     * @brief A class that generates a hash value for a given key as the probing step size.
     */
    class ProbeHash
    {
      public:
        /**
         * @brief Generate a hash value for a key as the probing step size.
         *
         * @param key the key.
         * @return std::size_t the hash value.
         */
        static std::size_t Generate(const Key &key)
        {
            return ((std::size_t)key & 0xff) * 7 + (((std::size_t)key >> 8) & 0xff) * 11 +
                   (((std::size_t)key >> 16) & 0xff) * 19;
        }
    };

    /**
     * @brief the sizes of the hash table.
     */
    static const std::size_t TableSizes[];

    /**
     * @brief the maximum number of pairs can be stored.
     */
    std::size_t size;

    /**
     * @brief the index of a size in TableSizes being the current size.
     */
    std::size_t sizeIndex;

    /**
     * @brief the current number of pairs being stored.
     */
    std::size_t nElements;

    /**
     * @brief the array used to store the key-value pairs.
     */
    Pair<Key, Value> *pairs;

    bool *insertMarks;

    bool *deleteMarks;

    void AllocateMemory()
    {
        pairs = new Pair<Key, Value>[size] {};
        insertMarks = new bool[size];
        deleteMarks = new bool[size];
        for (std::size_t i = 0; i < size; i++)
            insertMarks[i] = deleteMarks[i] = false;
    }

    void ReleaseMemory()
    {
        if (pairs)
            delete[] pairs;
        if (insertMarks)
            delete[] insertMarks;
        if (deleteMarks)
            delete[] deleteMarks;
    }

    void DynamicallyResize()
    {
        const float loadFactor = (float)nElements / size;
        if (loadFactor >= 0.75f)
            Expand();
        else if (loadFactor <= 0.3f)
            Shrink();
    }

    void Expand()
    {
        std::size_t oldSize = size;
        auto oldPairs = pairs;
        auto oldInsertMarks = insertMarks;
        auto oldDeleteMarks = deleteMarks;
        if (sizeIndex < sizeof(TableSizes) / sizeof(std::size_t))
        {
            sizeIndex++;
            size = TableSizes[sizeIndex];
            AllocateMemory();
        }
        else
        {
            size <<= 1;
            AllocateMemory();
        }
        for (std::size_t i = 0; i < oldSize; i++)
        {
            if (!oldInsertMarks[i] || oldDeleteMarks[i])
                continue;
            Insert(oldPairs[i].Key(), oldPairs[i].Value());
        }
        delete[] oldPairs;
        delete[] oldInsertMarks;
        delete[] oldDeleteMarks;
    }

    void Shrink()
    {
        if (sizeIndex == 0)
            return;
        std::size_t oldSize = size;
        auto oldPairs = pairs;
        auto oldInsertMarks = insertMarks;
        auto oldDeleteMarks = deleteMarks;
        if (sizeIndex < sizeof(TableSizes) / sizeof(std::size_t))
        {
            sizeIndex--;
            size = TableSizes[sizeIndex];
            AllocateMemory();
        }
        else if ((size >> 1) == TableSizes[sizeof(TableSizes) / sizeof(std::size_t) - 1])
        {
            sizeIndex = sizeof(TableSizes) / sizeof(std::size_t) - 1;
            size = TableSizes[sizeIndex];
            AllocateMemory();
        }
        else
        {
            size >>= 1;
            AllocateMemory();
        }
        for (std::size_t i = 0; i < oldSize; i++)
        {
            if (!oldInsertMarks[i] || oldDeleteMarks[i])
                continue;
            Insert(oldPairs[i].Key(), oldPairs[i].Value());
        }
        delete[] oldPairs;
        delete[] oldInsertMarks;
        delete[] oldDeleteMarks;
    }

    std::size_t FindPosition(const Key &key, bool search) const
    {
        const auto hashValue = KeyHash::Generate(key);
        auto currentIdx = hashValue % size;
        if (insertMarks[currentIdx] && !deleteMarks[currentIdx] && pairs[currentIdx].Key() == key)
            return currentIdx;
        if (!search && (!insertMarks[currentIdx] || deleteMarks[currentIdx]))
            return currentIdx;

        std::size_t i = 1;
        const auto stepSize = ProbeHash::Generate(key) % (size - 1) + 1;
        currentIdx = (hashValue + i * stepSize) % size;
        while (insertMarks[currentIdx])
        {
            if (deleteMarks[currentIdx] && !search)
                break;
            else if (!deleteMarks[currentIdx] && pairs[currentIdx].Key() == key)
                break;
            i++;
            currentIdx = (hashValue + i * stepSize) % size;
        }
        return currentIdx;
    }

  public:
    class Iterator
    {
      private:
        HashTable *owner;
        std::size_t idx;
        std::size_t startIdx;

        void FindNext()
        {
            if (!owner)
                return;
            if (idx < owner->size)
                idx++;
            while (idx < owner->size && (!owner->insertMarks[idx] || owner->deleteMarks[idx]))
                idx++;
        }

        void FindPrev()
        {
            if (!owner)
                return;
            if (idx > startIdx)
                idx--;
            while (idx > startIdx && (!owner->insertMarks[idx] || owner->deleteMarks[idx]))
                idx--;
        }

      public:
        Iterator(HashTable *pTable = nullptr, bool end = false) : owner(pTable), idx(0), startIdx(0)
        {
            if (!pTable)
                return;
            if (end)
            {
                idx = pTable->size;
                return;
            }
            while (idx < pTable->size && (!pTable->insertMarks[idx] || pTable->deleteMarks[idx]))
                idx++;
            startIdx = idx;
        }

        Pair<Key, Value> &operator*()
        {
            return owner->pairs[idx];
        }

        Pair<Key, Value> *operator->()
        {
            return &owner->pairs[idx];
        }

        HashTable<Key, Value>::Iterator &operator++()
        {
            FindNext();
            return *this;
        }

        HashTable<Key, Value>::Iterator operator++(int)
        {
            FindNext();
            return *this;
        }

        HashTable<Key, Value>::Iterator &operator--()
        {
            FindPrev();
            return *this;
        }

        HashTable<Key, Value>::Iterator operator--(int)
        {
            FindPrev();
            return *this;
        }

        bool operator==(const HashTable<Key, Value>::Iterator &other) const
        {
            return owner == other.owner && idx == other.idx;
        }

        bool operator!=(const HashTable<Key, Value>::Iterator &other) const
        {
            return owner != other.owner || idx != other.idx;
        }

        friend class HashTable<Key, Value>;
    };

    HashTable() : sizeIndex(0), nElements(0)
    {
        size = TableSizes[sizeIndex];
        AllocateMemory();
    }

    HashTable(const HashTable &other) : size(other.size), sizeIndex(other.sizeIndex), nElements(other.nElements)
    {
        AllocateMemory();
        for (std::size_t i = 0; i < other.size; i++)
        {
            if (!other.insertMarks[i] || other.deleteMarks[i])
                continue;
            Insert(other.pairs[i].Key(), other.pairs[i].Value());
        }
    }

    HashTable(HashTable &&other)
        : size(other.size), sizeIndex(other.sizeIndex), nElements(other.nElements), pairs(other.pairs),
          insertMarks(other.insertMarks), deleteMarks(other.deleteMarks)
    {
        other.size = 0;
        other.sizeIndex = 0;
        other.nElements = 0;
        other.pairs = nullptr;
        other.insertMarks = nullptr;
        other.deleteMarks = nullptr;
    }

    HashTable &operator=(const HashTable &other)
    {
        ReleaseMemory();
        size = other.size;
        sizeIndex = other.sizeIndex;
        nElements = other.nElements;
        AllocateMemory();
        for (std::size_t i = 0; i < other.size; i++)
        {
            if (!other.insertMarks[i] || other.deleteMarks[i])
                continue;
            Insert(other.pairs[i].Key(), other.pairs[i].Value());
        }
        return *this;
    }

    HashTable &operator=(HashTable &&other) noexcept
    {
        size = other.size;
        sizeIndex = other.sizeIndex;
        nElements = other.nElements;
        pairs = other.pairs;
        insertMarks = other.insertMarks;
        deleteMarks = other.deleteMarks;

        other.size = 0;
        other.sizeIndex = 0;
        other.nElements = 0;
        other.pairs = nullptr;
        other.insertMarks = nullptr;
        other.deleteMarks = nullptr;
        return *this;
    }

    ~HashTable()
    {
        ReleaseMemory();
    }

    template <class KeyType, class ValueType> void Insert(KeyType &&key, ValueType &&value)
    {
        const auto idx = FindPosition(key, false);
        insertMarks[idx] = true;
        deleteMarks[idx] = false;
        pairs[idx].Key() = Types::Forward<decltype(key)>(key);
        pairs[idx].Value() = Types::Forward<decltype(value)>(value);
        nElements++;
        DynamicallyResize();
    }

    void Remove(const Key &key)
    {
        if (IsEmpty())
            return;
        const auto idx = FindPosition(key, true);
        if (!insertMarks[idx] || deleteMarks[idx] || pairs[idx].Key() != key)
            return;
        deleteMarks[idx] = true;
        nElements--;
        DynamicallyResize();
    }

    void Clear()
    {
        *this = HashTable();
    }

    bool Contains(const Key &key) const
    {
        const auto idx = FindPosition(key, true);
        return !(!insertMarks[idx] || deleteMarks[idx] || pairs[idx].Key() != key);
    }

    HashTable<Key, Value>::Iterator Find(const Key &key) const
    {
        HashTable<Key, Value>::Iterator itr;
        itr.owner = (HashTable<Key, Value> *)this;
        const auto idx = FindPosition(key, true);
        if (insertMarks[idx] && !deleteMarks[idx] && pairs[idx].Key() == key)
            itr.idx = idx;
        else
            return Last();
        return itr;
    }

    bool IsEmpty() const
    {
        return nElements == 0;
    }

    std::size_t Length() const
    {
        return nElements;
    }

    HashTable<Key, Value>::Iterator First() const
    {
        return HashTable<Key, Value>::Iterator((HashTable<Key, Value> *)this, false);
    }

    HashTable<Key, Value>::Iterator Last() const
    {
        return HashTable<Key, Value>::Iterator((HashTable<Key, Value> *)this, true);
    }

    Value &operator[](const Key &key)
    {
        Iterator itr = Find(key);
        if (itr == Last())
        {
            Insert(key, Value());
            return Find(key)->Value();
        }
        return itr->Value();
    }

    const Value &operator[](const Key &key) const
    {
        return Find(key)->Value();
    }

    friend class HashTable<Key, Value>::Iterator;
};

template <class Key, class Value>
const std::size_t HashTable<Key, Value>::TableSizes[] = {13, 37, 79, 97, 199, 401, 857, 1699, 3307};
} // namespace ChiaData

#endif // HASH_TABLE_HPP
