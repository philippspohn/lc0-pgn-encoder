#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <regex>
#include "neural/encoder.h"
#include "chess/pgn.h"

struct Game {
    std::string whiteElo;
    std::string blackElo;
    std::vector<std::string> moves;
};

void printHelp() {
    std::cout << "Usage: pgn_processor [options] <PGN file path>\n"
              << "Options:\n"
              << "  -h, --help            Show this help message\n"
              << "  --discard-last        Optionally discard the last game in the file\n";
}

bool isValidElo(const std::string& elo) {
    return !elo.empty();
}

bool isValidGame(const Game& game) {
    return isValidElo(game.whiteElo) && isValidElo(game.blackElo) && game.moves.size() >= 10;
}

std::string trim(const std::string& str) {
    auto start = str.begin();
    while (start != str.end() && std::isspace(*start)) {
        start++;
    }
    auto end = str.end();
    do {
        end--;
    } while (std::distance(start, end) > 0 && std::isspace(*end));

    return {start, end + 1};
}

std::vector<Game> processPGNFile(const std::string& filePath, bool discardLast) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << '\n';
        exit(1);
    }

    std::vector<Game> games;
    std::string line;
    Game currentGame;
    bool inMovesSection = false, isEmptyLine = false;
    std::size_t sectionCount = 0, maxSections = std::numeric_limits<std::size_t>::max();

    if (discardLast) {
        while (std::getline(file, line)) {
            if (line.empty() && !isEmptyLine) {
                ++sectionCount;
                isEmptyLine = true;
            } else if (!line.empty()) {
                isEmptyLine = false;
            }
        }
        maxSections = (sectionCount - 1) / 2 * 2;
        file.clear();
        file.seekg(0, std::ios::beg);
        sectionCount = 0;
        isEmptyLine = false;
    }

    std::regex moveRegex(R"(\d+\.\s*(\S+))"), resultRegex(R"(^1-0$|^0-1$|^1/2-1/2$)");
    while (std::getline(file, line) && sectionCount < maxSections) {
        line = trim(line);
        if (line.empty()) {
            if (!isEmptyLine) {
                ++sectionCount;
                if (!currentGame.moves.empty() && inMovesSection && isValidGame(currentGame)) {
                    games.push_back(currentGame);
                    currentGame = Game();
                }
                inMovesSection = false;
            }
            isEmptyLine = true;
            continue;
        } else {
            isEmptyLine = false;
        }

        if (line[0] == '[') {
            std::istringstream iss(line);
            std::string tag, value;
            iss >> tag;
            std::getline(iss, value, '\"');
            std::getline(iss, value, '\"');
            if (tag == "[WhiteElo") currentGame.whiteElo = value;
            else if (tag == "[BlackElo") currentGame.blackElo = value;
        } else {
            inMovesSection = true;
            std::istringstream iss(line);
            std::string token;
            while (iss >> token) {
                if (std::regex_match(token, resultRegex)) {
                    break;
                }
                std::smatch matches;
                if (std::regex_search(token, matches, moveRegex) && matches.size() > 1) {
                    currentGame.moves.push_back(matches[1].str());
                } else if (token.find('.') == std::string::npos) {
                    currentGame.moves.push_back(token);
                }
            }
        }
    }

    if (!discardLast && isValidGame(currentGame)) {
        games.push_back(currentGame);
    }

    return games;
}
void encodeAndAppendInputPlanes(const lczero::InputPlanes& planes, std::ostringstream& stream) {
    for (auto it = planes.begin(); it != planes.end(); ++it) {
        const auto& plane = *it;
        stream << plane.mask << ',' << plane.value;
        if (std::next(it) != planes.end()) {
            stream << ';';
        }
    }
}

void encodeAndWriteGames(const std::vector<Game>& games, const std::string& outFileName) {
    std::ofstream outFile(outFileName);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open file for writing.\n";
        return;
    }

    for (const auto& game : games) {
        lczero::ChessBoard board;
        lczero::PositionHistory history;
        board.SetFromFen(lczero::ChessBoard::kStartposFen);
        history.Reset(board, 0, 1);

        std::ostringstream gameEncoding;
        gameEncoding << game.whiteElo << "," << game.blackElo << "|";

        bool isFirstPosition = true;
        for (const auto& moveStr : game.moves) {
            if (!isFirstPosition) {
                gameEncoding << "|";
            }
            isFirstPosition = false;

            lczero::Move move = lczero::SanToMove(moveStr, history.Last().GetBoard());
            history.Append(move);

            lczero::InputPlanes planes = lczero::EncodePositionForNN(
                    pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE, history, 8,
                    lczero::FillEmptyHistory::ALWAYS, nullptr
            );

            encodeAndAppendInputPlanes(planes, gameEncoding);
        }
        outFile << gameEncoding.str() << "\n";
    }

    std::cout << "Processed " << games.size() << " games.\n";
    outFile.close();
}

int main(int argc, char* argv[]) {
    if (argc < 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
        printHelp();
        return 1;
    }

    bool discardLast = false;
    std::string filePath;
    std::string outFileName = "input_planes.txt";

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--discard-last") {
            discardLast = true;
        } else if (std::string(argv[i]) == "--output" && i + 1 < argc) {
            outFileName = argv[++i];
        } else {
            filePath = argv[i];
        }
    }

    lczero::InitializeMagicBitboards();

    auto games = processPGNFile(filePath, discardLast);

    encodeAndWriteGames(games, outFileName);

    std::cout << "Processed " << games.size() << " games and output to " << outFileName << ".\n";
    return 0;
}