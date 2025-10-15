#include "SWsolver.hpp"
SWsolver::SWsolver(SequenceBatch& query, 
                   SequenceBatch& reference, 
                   std::int32_t match, std::int32_t mismatch, 
                   std::int32_t del, std::int32_t ins)
    : query(query), reference(reference), match(match), mismatch(mismatch), del(del), ins(ins), result(ResultsBatch(2*query.sequence_size(), query.num_batches()))
    {
        if (query.num_batches() != reference.num_batches())
        {
            throw std::invalid_argument("Query and reference must have the same number of sequences");
        }
        if (query.sequence_size() != reference.sequence_size())
        {
            throw std::invalid_argument("Query and reference sequences must have the same length");
        }
    }



SWsolver::SWsolver(SequenceBatch& query, 
                SequenceBatch& reference)
: query(query), reference(reference), match(-1), mismatch(1), del(2), ins(2), result(2*query.sequence_size(), query.num_batches()) {
    if (query.num_batches() != reference.num_batches())
    {
        throw std::invalid_argument("Query and reference must have the same number of sequences");
    }
    if (query.sequence_size() != reference.sequence_size())
    {
        throw std::invalid_argument("Query and reference sequences must have the same length");
    }
}


ResultsBatch SWsolver::getResult()
{
    if (!solved)
    {
        result = compute(); // MAKE SURE THIS HERE USES THE MOVE SEMANTICS
        solved = true;
    }
    return result;
}

ResultsBatch SWsolver::compute()
{
    // Smith-Waterman algorithm implementation
    std::size_t rows = query.sequence_size() + 1;
    std::size_t cols = reference.sequence_size() + 1;
    Matrix<std::int32_t> scoreMatrix(rows, std::vector<std::int32_t>(cols, 0));
    Matrix<Direction> directionMatrix(rows, std::vector<Direction>(cols, Direction::NONE));

    for (std::size_t k = 0; k < query.num_batches(); ++k)
    {
        auto query_k = query[k];
        auto reference_k = reference[k];
        std::int32_t maxScore = 0;
        std::size_t maxI = 0, maxJ = 0;

        for (std::size_t i = 1; i < rows; ++i)
        {
            for (std::size_t j = 1; j < cols; ++j)
            {
                std::int32_t scoreDiag = (query_k[i-1] == reference_k[j-1]) ? match : mismatch;
                scoreDiag += (i > 0 && j > 0) ? scoreMatrix[i - 1][j - 1] : 0;
                std::int32_t scoreUp = (i > 0) ? scoreMatrix[i - 1][j] - del : 0;
                std::int32_t scoreLeft = (j > 0) ? scoreMatrix[i][j - 1] - ins : 0;

                auto max4 = [](std::int32_t a, std::int32_t b, std::int32_t c, std::int32_t d)
                {
                    return std::max(a, std::max(b, std::max(c, d)));
                };
                scoreMatrix[i][j] = max4(0, scoreDiag, scoreUp, scoreLeft);

                if (scoreMatrix[i][j] == scoreDiag && scoreMatrix[i][j] != 0)
                {
                    directionMatrix[i][j] = Direction::DIAGONAL;
                }
                else if (scoreMatrix[i][j] == scoreUp && scoreMatrix[i][j] != 0)
                {
                    directionMatrix[i][j] = Direction::UP;
                }
                else if (scoreMatrix[i][j] == scoreLeft && scoreMatrix[i][j] != 0)
                {
                    directionMatrix[i][j] = Direction::LEFT;
                }
                else
                {
                    directionMatrix[i][j] = Direction::NONE;
                }

                if (scoreMatrix[i][j] > maxScore)
                {
                    maxScore = scoreMatrix[i][j];
                    maxI = i;
                    maxJ = j;
                }
            }
        }
        // Backtrace
        auto alignment = this->result[k];
        std::size_t i = maxI, j = maxJ, z = 0;
        while (i > 0 && j > 0 && scoreMatrix[i][j] != 0)
        {
            switch (directionMatrix[i][j])
            {
                case Direction::DIAGONAL:
                    alignment[z] = query_k[i - 1];
                    --i; --j;
                    break;
                case Direction::UP:
                    alignment[z] = query_k[i - 1];
                    --i;
                    break;
                case Direction::LEFT:
                    alignment[z] = reference_k[j - 1];
                    --j;
                    break;
                default:
                    break;
            }
            ++z;
        }

        std::reverse(alignment.begin(), alignment.begin() + z);
        this->result.setLength(k, z);
    }
    return result;
}
