#!/bin/bash
# Pronunciation Audit Script
# Plays each word's audio file so you can verify pronunciation.
# Usage: ./scripts/audit_pronunciation.sh [--flagged-only] [--start-from WORD]
#
# Controls:
#   Press ENTER to continue to the next word
#   Type 'r' + ENTER to replay the current word
#   Type 'b' + ENTER to flag the word as bad pronunciation
#   Type 'q' + ENTER to quit
#
# Flagged words are saved to scripts/pronunciation_flags.txt

VOCAB_DIR="ARYA/Resources/Vocabulary"
FLAGS_FILE="scripts/pronunciation_flags.txt"
FLAGGED_ONLY=false
START_FROM=""

# Known high-risk words for TTS mispronunciation
FLAGGED_WORDS=(
    "camel"       # KAM-ul, TTS sometimes says kuh-MEL
    "coconut"     # KOH-kuh-nut
    "couch"       # KOWCH
    "crayon"      # KRAY-on vs KRAN (regional)
    "cupboard"    # KUB-erd (silent p)
    "giraffe"     # juh-RAF
    "hippo"       # HIP-oh (short for hippopotamus)
    "jellyfish"   # JEL-ee-fish
    "monkey"      # MUNG-kee
    "octopus"     # OK-tuh-pus
    "orange"      # OR-inj (first syllable stress)
    "oven"        # UH-ven
    "parrot"      # PAIR-ut
    "penguin"     # PEN-gwin
    "picture"     # PIK-cher
    "pillow"      # PIL-oh
    "pineapple"   # PINE-ap-ul
    "scissors"    # SIZ-erz
    "squirrel"    # SKWIR-ul
    "toilet"      # TOY-let
    "umbrella"    # um-BREL-uh
    "watermelon"  # WAH-ter-mel-un
    "zebra"       # ZEE-bruh (American) - already fixed
)

while [[ $# -gt 0 ]]; do
    case $1 in
        --flagged-only) FLAGGED_ONLY=true; shift ;;
        --start-from) START_FROM="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Initialize flags file if it doesn't exist
touch "$FLAGS_FILE"

# Get sorted list of words
words=()
for dir in "$VOCAB_DIR"/*/; do
    word=$(basename "$dir")
    [ "$word" = "_prompts" ] && continue
    words+=("$word")
done
IFS=$'\n' words=($(sort <<<"${words[*]}")); unset IFS

total=${#words[@]}
flagged_count=0
reviewed_count=0
started=false

if [ -z "$START_FROM" ]; then
    started=true
fi

echo "=== PRONUNCIATION AUDIT ==="
echo "Total words: $total"
echo "Controls: ENTER=next, r=replay, b=flag bad, q=quit"
echo ""

for i in "${!words[@]}"; do
    word="${words[$i]}"

    # Skip until we reach start word
    if [ "$started" = false ]; then
        if [ "$word" = "$START_FROM" ]; then
            started=true
        else
            continue
        fi
    fi

    # If flagged-only mode, skip unflagged words
    if [ "$FLAGGED_ONLY" = true ]; then
        is_flagged=false
        for fw in "${FLAGGED_WORDS[@]}"; do
            [ "$word" = "$fw" ] && is_flagged=true
        done
        grep -q "^$word$" "$FLAGS_FILE" 2>/dev/null && is_flagged=true
        [ "$is_flagged" = false ] && continue
    fi

    audio="$VOCAB_DIR/$word/audio.m4a"
    num=$((i + 1))

    # Check if already flagged
    already_flagged=""
    grep -q "^$word$" "$FLAGS_FILE" 2>/dev/null && already_flagged=" [FLAGGED]"

    echo -n "[$num/$total] $word$already_flagged — playing... "
    afplay "$audio" 2>/dev/null

    while true; do
        read -r -p "(enter/r/b/q): " cmd
        case $cmd in
            r|R)
                echo -n "  Replaying '$word'... "
                afplay "$audio" 2>/dev/null
                ;;
            b|B)
                if ! grep -q "^$word$" "$FLAGS_FILE" 2>/dev/null; then
                    echo "$word" >> "$FLAGS_FILE"
                fi
                echo "  Flagged '$word' as bad pronunciation"
                flagged_count=$((flagged_count + 1))
                break
                ;;
            q|Q)
                echo ""
                echo "=== AUDIT PAUSED ==="
                echo "Reviewed: $reviewed_count words"
                echo "Flagged this session: $flagged_count"
                echo "Total flagged: $(wc -l < "$FLAGS_FILE" | tr -d ' ')"
                echo "Resume with: ./scripts/audit_pronunciation.sh --start-from '$word'"
                exit 0
                ;;
            *)
                break
                ;;
        esac
    done
    reviewed_count=$((reviewed_count + 1))
done

echo ""
echo "=== AUDIT COMPLETE ==="
echo "Reviewed: $reviewed_count words"
echo "Flagged this session: $flagged_count"
echo "Total flagged: $(wc -l < "$FLAGS_FILE" | tr -d ' ')"
if [ -s "$FLAGS_FILE" ]; then
    echo ""
    echo "Flagged words (in $FLAGS_FILE):"
    sort "$FLAGS_FILE" | while read -r w; do echo "  - $w"; done
fi
