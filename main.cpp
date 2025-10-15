#include "SWsolver.hpp"
#include <iostream>
#include <vector>
#include <ctime>

int main() //TODO: IMPLEMENT TESTS AGAINST EDGE CASES (strongly mismatching sequences, very short sequences, very long sequences, empty sequences, etc.)
{
    const std::size_t S_LEN = 256;
    const std::size_t N = 1024;

    std::cout << "Starting Smith-Waterman solver..." << std::endl;
    auto query = SequenceBatch(S_LEN, N);
    auto reference = SequenceBatch(S_LEN, N);
    SWsolver solver(query, reference);
    auto res = solver.getResult();

    constexpr size_t seq = 100;
    auto alignment_span = std::span<Aminoacid>(res[seq].data(), res.getLength(seq));
    std::cout << "First alignment (length " << res.getLength(seq) << "): ";
    for (const auto& aa : alignment_span)
    {
        std::cout << aa;
    }
    std::cout << std::endl;

    return 0;
}