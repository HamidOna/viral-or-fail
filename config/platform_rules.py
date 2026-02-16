"""
Platform-specific scoring rubrics for the Algorithm Simulator agent.

Each platform has weighted criteria that reflect how its recommendation
algorithm actually prioritises content. Weights sum to 1.0 (100%).
"""

PLATFORMS = ["TikTok", "Twitter/X", "YouTube", "Instagram"]

PLATFORM_RULES = {
    "TikTok": {
        "description": "Short-form video platform driven by the For You Page algorithm",
        "format_hint": "Short-form video (15-60s), vertical, with trending audio",
        "criteria": {
            "hook_strength": {
                "weight": 0.30,
                "description": (
                    "How strong is the opening hook? TikTok's algorithm measures "
                    "retention in the first 1-3 seconds. A weak hook kills distribution."
                ),
            },
            "trend_alignment": {
                "weight": 0.25,
                "description": (
                    "Does the content ride a current trend, sound, or format? "
                    "The FYP algorithm boosts content that matches trending signals."
                ),
            },
            "shareability": {
                "weight": 0.20,
                "description": (
                    "Would viewers send this to a friend or duet/stitch it? "
                    "Shares are the highest-weight engagement signal on TikTok."
                ),
            },
            "hashtag_strategy": {
                "weight": 0.15,
                "description": (
                    "Are hashtags relevant and discoverable? Mixing niche and broad "
                    "hashtags helps the algorithm classify and distribute content."
                ),
            },
            "audio_reference": {
                "weight": 0.10,
                "description": (
                    "Does the post reference or suggest a trending audio? "
                    "TikTok's algorithm clusters content by audio for distribution."
                ),
            },
        },
    },
    "Twitter/X": {
        "description": "Text-first microblogging platform driven by engagement velocity",
        "format_hint": "Tweet or thread, hot takes, quote-retweet bait",
        "criteria": {
            "hot_take_factor": {
                "weight": 0.30,
                "description": (
                    "Does the post have a strong, polarising, or surprising opinion? "
                    "Twitter/X rewards engagement velocity — hot takes drive replies."
                ),
            },
            "quote_retweet_bait": {
                "weight": 0.25,
                "description": (
                    "Is the post structured to invite quote retweets? QRTs are "
                    "Twitter/X's most powerful distribution mechanic."
                ),
            },
            "timing_relevance": {
                "weight": 0.20,
                "description": (
                    "Is this posted at the right moment? Twitter/X's algorithm "
                    "heavily weights recency and real-time relevance."
                ),
            },
            "thread_potential": {
                "weight": 0.15,
                "description": (
                    "Could this expand into a thread? Threads increase time-on-post "
                    "and signal depth to the algorithm."
                ),
            },
            "hashtag_strategy": {
                "weight": 0.10,
                "description": (
                    "Are hashtags used sparingly and strategically? Over-hashtagging "
                    "on Twitter/X reduces credibility and reach."
                ),
            },
        },
    },
    "YouTube": {
        "description": "Long and short-form video platform driven by watch time and CTR",
        "format_hint": "Video (Shorts or long-form), strong thumbnail + title",
        "criteria": {
            "thumbnail_clickability": {
                "weight": 0.25,
                "description": (
                    "Would the suggested thumbnail stop a scroll? YouTube's algorithm "
                    "uses click-through rate as a primary ranking signal."
                ),
            },
            "title_curiosity_gap": {
                "weight": 0.25,
                "description": (
                    "Does the title create curiosity without being pure clickbait? "
                    "The algorithm balances CTR against viewer satisfaction."
                ),
            },
            "watch_time_potential": {
                "weight": 0.20,
                "description": (
                    "Will viewers watch most of the video? Average view duration "
                    "is YouTube's strongest ranking signal."
                ),
            },
            "seo_optimization": {
                "weight": 0.15,
                "description": (
                    "Are keywords, tags, and description optimised for search? "
                    "YouTube is the world's second-largest search engine."
                ),
            },
            "community_engagement": {
                "weight": 0.15,
                "description": (
                    "Does the content encourage comments and likes? Engagement rate "
                    "signals quality to YouTube's recommendation system."
                ),
            },
        },
    },
    "Instagram": {
        "description": "Visual-first platform driven by saves, shares, and Explore page",
        "format_hint": "Reel, carousel, or single image with strong caption",
        "criteria": {
            "visual_appeal": {
                "weight": 0.30,
                "description": (
                    "Is the visual content eye-catching and high quality? Instagram's "
                    "algorithm prioritises posts that generate saves and long views."
                ),
            },
            "caption_hook": {
                "weight": 0.20,
                "description": (
                    "Does the caption hook in the first line (before 'more')? "
                    "Caption engagement drives the algorithm's interest scoring."
                ),
            },
            "carousel_potential": {
                "weight": 0.20,
                "description": (
                    "Could this be a swipeable carousel? Carousels get re-served "
                    "by the algorithm when users didn't swipe through all slides."
                ),
            },
            "hashtag_reach": {
                "weight": 0.15,
                "description": (
                    "Is there a mix of niche and popular hashtags (20-30 range)? "
                    "Instagram uses hashtags to classify content for Explore."
                ),
            },
            "story_crosspost": {
                "weight": 0.15,
                "description": (
                    "Is this structured to also work as a Story or Reel? "
                    "Cross-format posting increases total distribution surface."
                ),
            },
        },
    },
}
