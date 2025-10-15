#include <vector>
#include <cstdint>
#include <algorithm>
#include <iostream>
#include <span>

template <typename T>
using Matrix = std::vector<std::vector<T>>;

enum class Aminoacid
{
    T,
    A,
    C,
    G,
    N
};

inline std::ostream& operator<<(std::ostream& os, const Aminoacid& obj)
{
    switch (obj)
    {
        case Aminoacid::T: os << 'T'; break;
        case Aminoacid::A: os << 'A'; break;
        case Aminoacid::C: os << 'C'; break;
        case Aminoacid::G: os << 'G'; break;
        case Aminoacid::N: os << 'N'; break;
    }
    return os;
}

class SequenceBatch
{
public:
    SequenceBatch(std::size_t seq_size, std::size_t num_sequences) 
    {
        data = std::vector<Aminoacid>(seq_size * num_sequences);
        offsets = std::vector<size_t>(num_sequences + 1);
        this->seq_size = seq_size;
        this->num_sequences = num_sequences;
        
        for (size_t i = 0; i < offsets.size(); i++)
        {
            offsets[i] = i * seq_size;
        }
        for (size_t i = 0; i < data.size(); i++)
        {
            data[i] = static_cast<Aminoacid>(std::rand() % 5);
        }
    }

    SequenceBatch(std::vector<Aminoacid>&& data, std::vector<size_t>&& offsets) 
        : data(std::move(data)), offsets(std::move(offsets)) 
    {
        if (offsets.size() < 2 || offsets[0] != 0 || offsets.back() != data.size())
        {
            throw std::invalid_argument("Invalid offsets");
        }
    }

    size_t num_batches() const
    {
        return num_sequences;
    }
    size_t sequence_size() const
    {
        return seq_size;
    }
    std::span<Aminoacid> operator[](size_t i)
    {
        return std::span<Aminoacid>(data.data() + offsets[i], offsets[i + 1] - offsets[i]);
    }

private:
    std::vector<Aminoacid> data;
    std::vector<size_t> offsets;
    std::size_t seq_size;
    std::size_t num_sequences;
};

class ResultsBatch : public SequenceBatch
{
public:
    ResultsBatch(std::size_t seq_size, std::size_t num_sequences)
        : SequenceBatch(seq_size, num_sequences), lengths(num_sequences, 0)
    {}

    size_t getLength(size_t i) const
    {
        return lengths[i];
    }

    void setLength(size_t i, size_t length)
    {
        if (length > sequence_size())
        {
            throw std::out_of_range("Length exceeds sequence size");
        }
        lengths[i] = length;
    }
private:
    std::vector<size_t> lengths;

};

enum class Direction
{
    NONE = 0,
    DIAGONAL = 1,
    UP = 2,
    LEFT = 3
};

class SWsolver //TODO: make this class use shared pointers for the sequences
{
public:
    // Takes vectors as lvalues
    SWsolver(SequenceBatch& query, SequenceBatch& reference, std::int32_t match, std::int32_t mismatch, std::int32_t del, std::int32_t ins);

    // Default initialization of parameters
    SWsolver(SequenceBatch& query, SequenceBatch& reference);


    ResultsBatch getResult();

private:
    SequenceBatch& query;
    SequenceBatch& reference;
    ResultsBatch result;
    bool solved = false;

    const std::int32_t match;
    const std::int32_t mismatch;
    const std::int32_t del;
    const std::int32_t ins;

    ResultsBatch compute();
};