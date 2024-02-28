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

bool isValidElo(const std::string &elo) {
    return !elo.empty();
}

bool isValidGame(const Game &game) {
    return isValidElo(game.whiteElo) && isValidElo(game.blackElo) && game.moves.size() >= 10;
}

std::string trim(const std::string &str) {
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

std::vector<Game> processPGNFile(const std::string &filePath, bool discardLast) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << '\n';
        exit(1);
    }

    std::vector<Game> games;
    std::size_t skippedGames = 0;
    std::string line, processedLine;
    Game currentGame;
    bool inMovesSection = false, isEmptyLine = false, inComment = false;
    std::size_t sectionCount = 0, maxSections = std::numeric_limits<std::size_t>::max();

    if (discardLast) {
        skippedGames++;
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

    std::string results[] = {"1-0", "0-1", "1/2-1/2"};
    while (std::getline(file, line) && sectionCount < maxSections) {
        processedLine = "";
        for (char ch: line) {
            if (!inComment && ch == '{') {
                inComment = true;
                continue;
            }
            if (inComment && ch == '}') {
                inComment = false;
                continue;
            }
            if (!inComment) {
                processedLine += ch;
            }
        }

        processedLine = trim(processedLine);
        if (trim(line).empty()) {
            if (!isEmptyLine) {
                ++sectionCount;
                if (!currentGame.moves.empty() && inMovesSection) {
                    if(isValidGame(currentGame)) {
                        games.push_back(currentGame);
                    } else {
                        skippedGames++;
                    }
                    currentGame = Game();
                    if (games.size() % 500 == 0)
                        std::cout << "\rLoaded " << games.size() << " games." << std::flush;
                }
                inMovesSection = false;
            }
            isEmptyLine = true;
            continue;
        } else {
            isEmptyLine = false;
        }

        if (processedLine[0] == '[') {
            std::istringstream iss(processedLine);
            std::string tag, value;
            iss >> tag;
            std::getline(iss, value, '\"');
            std::getline(iss, value, '\"');
            if (tag == "[WhiteElo") currentGame.whiteElo = value;
            else if (tag == "[BlackElo") currentGame.blackElo = value;
        } else {
            inMovesSection = true;
            std::istringstream iss(processedLine);
            std::string token;
            while (iss >> token) {
                // Check for game result
                bool isResult = false;
                for (const auto &result: results) {
                    if (token == result) {
                        isResult = true;
                        break;
                    }
                }
                if (isResult) break;

                // Check move
                size_t dotPos = token.find('.');
                if (dotPos != std::string::npos) {
                    size_t startPos = (token.find("...") != std::string::npos) ? dotPos + 3 : dotPos + 1;
                    std::string move = token.substr(startPos);
                    if (!move.empty()) currentGame.moves.push_back(move);
                } else {
                    currentGame.moves.push_back(token);
                }
            }
        }
    }

    if (!discardLast && isValidGame(currentGame)) {
        games.push_back(currentGame);
    }

    std::cout << "\rCompleted - Loaded " << games.size() << "/" << games.size() + skippedGames << " games.\n";

    return games;
}

void encodeAndAppendInputPlanes(const lczero::InputPlanes &planes, std::string &stream) {
    for (auto it = planes.begin(); it != planes.end(); ++it) {
        const auto &plane = *it;
        stream += plane.mask + ',' + plane.value;
        if (std::next(it) != planes.end()) {
            stream += ';';
        }
    }
}

void encodeAndWriteGames(const std::vector<Game> &games, const std::string &outFileName) {
    std::ofstream outFile(outFileName);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open file for writing.\n";
        return;
    }

    size_t gamesProcessed = 0;
    for (const auto &game: games) {
        lczero::ChessBoard board;
        lczero::PositionHistory history;
        board.SetFromFen(lczero::ChessBoard::kStartposFen);
        history.Reset(board, 0, 1);

        std::string gameEncoding;
        gameEncoding += game.whiteElo + "," + game.blackElo + "|";
        std::string gameString;

        bool isFirstPosition = true;
        bool skipGame = false;
        for (const auto &moveStr: game.moves) {
            if (!isFirstPosition) {
                gameEncoding += "|";
            }
            isFirstPosition = false;
            gameString += moveStr + " ";
            try {
                lczero::Move move = lczero::SanToMove(moveStr, history.Last().GetBoard());
                history.Append(move);
            } catch (...) {
                std::cerr << "Illegal move (" << moveStr << "), skipping game!\n";
                skipGame = true;
                break;
            }

            lczero::InputPlanes planes = lczero::EncodePositionForNN(
                    pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE, history, 8,
                    lczero::FillEmptyHistory::ALWAYS, nullptr
            );

            encodeAndAppendInputPlanes(planes, gameEncoding);
        }
        if (!skipGame) {
            outFile << gameEncoding << "\n";
        }
        gamesProcessed++;
        if (gamesProcessed % 250 == 0) {
            std::cout << "\rProcessed " << gamesProcessed << "/" << games.size() << " games." << std::flush;
        }
    }

    std::cout << "\rProcessed " << games.size() << " games.\n";
    std::cout << "Wrote output to " << outFileName << ".\n";
    outFile.close();
}

int main(int argc, char *argv[]) {
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

    std::cout << "Reading PGN file...\n";
    auto games = processPGNFile(filePath, discardLast);

    std::cout << "Generating and writing input planes...\n";
    encodeAndWriteGames(games, outFileName);
    return 0;
}