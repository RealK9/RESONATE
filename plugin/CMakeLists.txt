cmake_minimum_required(VERSION 3.22)

project(ResonateBridge VERSION 1.0.0)

# ── JUCE ──
# Download JUCE via FetchContent if not found locally
include(FetchContent)
FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG        8.0.4
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(JUCE)

# ── Plugin Target ──
juce_add_plugin(ResonateBridge
    COMPANY_NAME           "RESONATE"
    PLUGIN_MANUFACTURER_CODE  Rsnt
    PLUGIN_CODE               RsBr
    FORMATS                   VST3 AU Standalone
    PRODUCT_NAME              "RESONATE Bridge"
    IS_SYNTH                  FALSE
    NEEDS_MIDI_INPUT          FALSE
    NEEDS_MIDI_OUTPUT         FALSE
    IS_MIDI_EFFECT            FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD   FALSE
)

juce_generate_juce_header(ResonateBridge)

target_sources(ResonateBridge
    PRIVATE
        Source/PluginProcessor.cpp
        Source/PluginProcessor.h
        Source/PluginEditor.cpp
        Source/PluginEditor.h
        Source/BridgeClient.cpp
        Source/BridgeClient.h
)

target_compile_definitions(ResonateBridge
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
        JUCE_DISPLAY_SPLASH_SCREEN=0
)

target_link_libraries(ResonateBridge
    PRIVATE
        juce::juce_audio_utils
        juce::juce_audio_processors
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)
