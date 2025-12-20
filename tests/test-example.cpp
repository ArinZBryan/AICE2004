#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace Catch::Matchers;

TEST_CASE("Example Test", "[Example.cpp]") {
    REQUIRE(1 == 1);
}
