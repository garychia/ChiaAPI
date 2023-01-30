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

    /**
     * @brief indices whether a location has been inserted or not.
     */
    bool *insertMarks;

    /**
     * @brief indices whether a location has been deleted or not.
     */
    bool *deleteMarks;

    /**
     * @brief Allocate memory for the buffers.
     */
    void AllocateMemory()
    {
        pairs = new Pair<Key, Value>[size] {};
        insertMarks = new bool[size];
        deleteMarks = new bool[size];
        for (std::size_t i = 0; i < size; i++)
            insertMarks[i] = deleteMarks[i] = false;
    }

    /**
     * @brief Release the buffers.
     */
    void ReleaseMemory()
    {
        if (pairs)
            delete[] pairs;
        if (insertMarks)
            delete[] insertMarks;
        if (deleteMarks)
            delete[] deleteMarks;
    }

    /**
     * @brief Resize the hash table dynamically.
     */
    void DynamicallyResize()
    {
        const float loadFactor = (float)nElements / size;
        if (loadFactor >= 0.75f)
            Expand();
        else if (loadFactor <= 0.3f)
            Shrink();
    }

    /**
     * @brief Expand the hash table to store more values.
     */
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

    /**
     * @brief Shrink the hash table if it is sparse.
     */
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

    /**
     * @brief Find a position of the key-value pair.
     *
     * @param key the key of the pair.
     * @param search true if the key is used to search for a location. Otherwise, it is for the insertion.
     * @return std::size_t the position of the key-value pair found or to be inserted.
     */
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
        /**
         * @brief the HashTable this iterator refers to.
         */
        HashTable *owner;

        /**
         * @brief index of the entry in the HashTable this iterator points to.
         */
        std::size_t idx;

        /**
         * @brief index of the first entry in the HashTable.
         */
        std::size_t startIdx;

        /**
         * @brief Find the index of the next entry in the HashTable.
         */
        void FindNext()
        {
            if (!owner)
                return;
            if (idx < owner->size)
                idx++;
            while (idx < owner->size && (!owner->insertMarks[idx] || owner->deleteMarks[idx]))
                idx++;
        }

        /**
         * @brief Find the index of the previous entry in the HashTable.
         */
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
        /**
         * @brief Construct a new HashTable::Iterator object
         *
         * @param pTable the HashTable the iterator will refer to.
         * @param end whether this iterator will point to the end of the HashTable.
         */
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

        /**
         * @brief Retrieve the key-value pair this iterator points to.
         *
         * @return Pair<Key, Value>& the key-value pair.
         */
        Pair<Key, Value> &operator*()
        {
            return owner->pairs[idx];
        }

        /**
         * @brief Retrieve a member or function of the key-value pair this iterator points to.
         *
         * @return Pair<Key, Value>* the key-value pair.
         */
        Pair<Key, Value> *operator->()
        {
            return &owner->pairs[idx];
        }

        /**
         * @brief Make this iterator point to the next entry in the HashTable.
         *
         * @return HashTable<Key, Value>::Iterator& the iterator.
         */
        HashTable<Key, Value>::Iterator &operator++()
        {
            FindNext();
            return *this;
        }

        /**
         * @brief Make this iterator point to the next entry in the HashTable.
         *
         * @return HashTable<Key, Value>::Iterator the iterator.
         */
        HashTable<Key, Value>::Iterator operator++(int)
        {
            FindNext();
            return *this;
        }

        /**
         * @brief Make this iterator point to the previous entry in the HashTable.
         *
         * @return HashTable<Key, Value>::Iterator the iterator.
         */
        HashTable<Key, Value>::Iterator &operator--()
        {
            FindPrev();
            return *this;
        }

        /**
         * @brief Make this iterator point to the next entry in the HashTable.
         *
         * @return HashTable<Key, Value>::Iterator the iterator.
         */
        HashTable<Key, Value>::Iterator operator--(int)
        {
            FindPrev();
            return *this;
        }

        /**
         * @brief Check if two iterators point to the same entry in the same HashTable.
         *
         * @param other another iterator.
         * @return true if they point to the same entry in the same HashTable.
         * @return false otherwise.
         */
        bool operator==(const HashTable<Key, Value>::Iterator &other) const
        {
            return owner && owner == other.owner && idx == other.idx;
        }

        /**
         * @brief Check if two iterators point to different entries or different HashTables.
         *
         * @param other another iterator.
         * @return true if they point to different entries or different HashTables.
         * @return false otherwise.
         */
        bool operator!=(const HashTable<Key, Value>::Iterator &other) const
        {
            return !operator==(other);
        }

        friend class HashTable<Key, Value>;
    };

    /**
     * @brief Construct a new HashTable object.
     */
    HashTable() : sizeIndex(0), nElements(0)
    {
        size = TableSizes[sizeIndex];
        AllocateMemory();
    }

    /**
     * @brief Construct a new HashTable object by copying an existing HashTable.
     *
     * @param other a HashTable to be copied.
     */
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

    /**
     * @brief Construct a new HashTable object.
     *
     * @param other a HashTable to be 'moved'.
     */
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

    /**
     * @brief Copy a HashTable.
     *
     * @param other a HashTable to be copied.
     * @return HashTable& this HashTable.
     */
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

    /**
     * @brief Move a HashTable.
     *
     * @param other a HashTable to be 'moved'.
     * @return HashTable& this HashTable.
     */
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

    /**
     * @brief Destroy the HashTable object.
     */
    ~HashTable()
    {
        ReleaseMemory();
    }

    /**
     * @brief Insert a key-value pair into the HashTable.
     *
     * @tparam KeyType the type of key.
     * @tparam ValueType the type of value.
     * @param key the key.
     * @param value the value.
     */
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

    /**
     * @brief Remove a key-value pair.
     *
     * @param key the key of the key-value pair to be removed.
     */
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

    /**
     * @brief Remove all the key-value pairs in the HashTable.
     */
    void Clear()
    {
        *this = HashTable();
    }

    /**
     * @brief Check if a key-value pair is present in the HashTable.
     *
     * @param key the key of the key-value pair to find.
     * @return true if the key-value pair is found.
     * @return false otherwise.
     */
    bool Contains(const Key &key) const
    {
        const auto idx = FindPosition(key, true);
        return !(!insertMarks[idx] || deleteMarks[idx] || pairs[idx].Key() != key);
    }

    /**
     * @brief Find a key-value pair.
     *
     * @param key the key of the key-value pair.
     * @return HashTable<Key, Value>::Iterator the iterator that points to the key-value pair if it is found. Otherwise,
     * HashTable::Last() is returned.
     */
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

    /**
     * @brief Check if the HashTable is empty.
     *
     * @return true if there is no key-value pair in the HashTable.
     * @return false otherwise.
     */
    bool IsEmpty() const
    {
        return nElements == 0;
    }

    /**
     * @brief Retrieve the number of key-value pairs in the HashTable.
     *
     * @return std::size_t the number of key-value pairs in the HashTable.
     */
    std::size_t Length() const
    {
        return nElements;
    }

    /**
     * @brief Get the iterator that points to the first key-value pair.
     *
     * @return HashTable<Key, Value>::Iterator the iterator that points to the first key-value pair.
     */
    HashTable<Key, Value>::Iterator First() const
    {
        return HashTable<Key, Value>::Iterator((HashTable<Key, Value> *)this, false);
    }

    /**
     * @brief Get the iterator that points to the end of the HashTable.
     *
     * @return HashTable<Key, Value>::Iterator the iterator that points to the end of the HashTable.
     */
    HashTable<Key, Value>::Iterator Last() const
    {
        return HashTable<Key, Value>::Iterator((HashTable<Key, Value> *)this, true);
    }

    /**
     * @brief Retrieve the value of a key-value pair.
     * 
     * @param key the key of the key-value pair.
     * @return Value& the value of the key-value pair.
     */
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

    /**
     * @brief Retrieve the value of a key-value pair.
     * 
     * @param key the key of the key-value pair.
     * @return const Value& the value of the key-value pair.
     */
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
