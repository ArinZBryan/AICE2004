#pragma once

#include <stdexcept>
#include <string>

#include "Train.h"

// Returns 0 on success, 1 on parse/validation error, 2 for help requested.
// Throws std::runtime_error on parse/validation error, or std::logic_error for help requested.
TrainConfig parse_arguments(int argc, char* argv[]);
