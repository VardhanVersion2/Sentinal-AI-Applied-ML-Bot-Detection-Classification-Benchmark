# Dataset Inventory & Specifications

## Cresci-2017 Benchmark Dataset

The Cresci-2017 dataset is a standard benchmark dataset for social bot and spambot detection on Twitter (X).

### Directory Placement
Extract or place your Cresci-2017 dataset folders inside:
```
data/raw/cresci2017/
├── genuine_accounts/
│   ├── users.csv
│   └── tweets.csv
├── social_spambots_1/
│   ├── users.csv
│   └── tweets.csv
├── social_spambots_2/
│   ├── users.csv
│   └── tweets.csv
├── social_spambots_3/
│   ├── users.csv
│   └── tweets.csv
├── traditional_spambots_1/
│   ├── users.csv
│   └── tweets.csv
└── fake_followers/
    ├── users.csv
    └── tweets.csv
```

### Class Mapping
| Directory | Category | Target Label (`is_bot`) | Description |
| :--- | :--- | :--- | :--- |
| `genuine_accounts` | Human / Genuine | `0` | Human-operated verified/genuine Twitter accounts |
| `social_spambots_1` | Social Spambot #1 | `1` | Retweeters of Italian political candidate |
| `social_spambots_2` | Social Spambot #2 | `1` | Spambots advertising mobile apps / goods |
| `social_spambots_3` | Social Spambot #3 | `1` | Spambots advertising Amazon products |
| `traditional_spambots_1`| Traditional Spambot | `1` | Traditional spam tweets containing spam URLs |
| `fake_followers` | Fake Followers | `1` | Inactive/purchased accounts to inflate follower counts |

### Primary User Attributes
- `id`: Unique user identifier
- `name`, `screen_name`: Display name and handle
- `statuses_count`: Total number of tweets posted
- `followers_count`: Followers
- `friends_count`: Followings / Friends
- `favourites_count`: Likes count
- `listed_count`: Number of public lists user belongs to
- `created_at`: Account registration timestamp
- `default_profile_image`: Flag indicating default avatar

