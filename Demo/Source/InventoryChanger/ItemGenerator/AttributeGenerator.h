#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <random>

#include <InventoryChanger/Inventory/Structs.h>
#include <SDK/Constants/PaintkitConditionChances.h>

#include "ItemGenerator.h"
#include "TournamentMatches.h"
#include "Utils.h"

namespace inventory_changer::item_generator
{

template <typename RandomEngine>
class AttributeGenerator {
public:
    explicit AttributeGenerator(RandomEngine& randomEngine) : randomEngine{ randomEngine } {}

    [[nodiscard]] float generatePaintKitWear() const
    {
        using namespace csgo::paintkit_condition_chances;

        static constexpr auto wearRanges = std::to_array<float>({ 0.0f, 0.07f, 0.15f, 0.38f, 0.45f, 1.0f });
        static constexpr auto conditionChances = std::to_array<float>({ factoryNewChance, minimalWearChance, fieldTestedChance, wellWornChance, battleScarredChance });

        return randomEngine(std::piecewise_constant_distribution<float>{ wearRanges.begin(), wearRanges.end(), conditionChances.begin() });
    }

    [[nodiscard]] float generateFactoryNewPaintKitWear() const
    {
        return randomEngine(std::uniform_real_distribution<float>{ 0.0f, 0.07f });
    }

    [[nodiscard]] int generatePaintKitSeed() const
    {
        return randomEngine(std::uniform_int_distribution<>{ 1, 1000 });
    }

    [[nodiscard]] std::uint32_t generateServiceMedalIssueDate(std::uint16_t year) const
    {
        return getRandomDateTimestampOfYear(year);
    }

    [[nodiscard]] inventory::SouvenirPackage generateSouvenirPackage(csgo::Tournament tournament, TournamentMap map) const
    {
        return std::visit([this](const auto& matches) {
            inventory::SouvenirPackage souvenirPackage;

            if (matches.empty())
                return souvenirPackage;

            const auto& randomMatch = matches[randomEngine(std::uniform_int_distribution<std::size_t>{ 0, matches.size() - 1 })];
            souvenirPackage.tournamentStage = randomMatch.stage;
            souvenirPackage.tournamentTeam1 = randomMatch.team1;
            souvenirPackage.tournamentTeam2 = randomMatch.team2;

            if constexpr (std::is_same_v<decltype(randomMatch), const MatchWithMVPs&>) {
                if (const auto numberOfMVPs = countMVPs(randomMatch); numberOfMVPs > 0)
                    souvenirPackage.proPlayer = randomMatch.mvpPlayers[randomEngine(std::uniform_int_distribution<std::size_t>{ 0, numberOfMVPs - 1 })];
            }

            return souvenirPackage;
        }, getTournamentMatchesOnMap(tournament, map));
    }

    void shuffleStickers(std::uint8_t numberOfSlots, inventory::SkinStickers& stickers) const
    {
        assert(numberOfSlots <= stickers.size());
        std::shuffle(stickers.begin(), stickers.begin() + numberOfSlots, randomEngine);
    }

    [[nodiscard]] bool generateStatTrak() const
    {
        return randomEngine(std::uniform_int_distribution<>{ 0, 9 }) == 0;
    }

private:
    [[nodiscard]] static std::pair<std::time_t, std::time_t> clampTimespanToNow(std::time_t min, std::time_t max) noexcept
    {
        const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        return std::make_pair((std::min)(min, now), (std::min)(max, now));
    }

    [[nodiscard]] std::uint32_t getRandomDateTimestampOfYear(std::uint16_t year) const noexcept
    {
        const auto [min, max] = clampTimespanToNow(getStartOfYearTimestamp(year), getEndOfYearTimestamp(year));
        return static_cast<std::uint32_t>(randomEngine(std::uniform_int_distribution<long long>{ min, max }));
    }

    RandomEngine& randomEngine;
};

}
