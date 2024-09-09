{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "89819320",
   "metadata": {
    "papermill": {
     "duration": 0.014481,
     "end_time": "2023-01-07T07:06:50.967425",
     "exception": false,
     "start_time": "2023-01-07T07:06:50.952944",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Food Recommendation System\n",
    "This dataset represents the data related to food recommender system. Two datasets are included in this dataset file. First includes the dataset related to the foods, ingredients, cuisines involved. Second, includes the dataset of the rating system for the recommendation system.\n",
    "\n",
    "Kaggle Link: https://www.kaggle.com/datasets/schemersays/food-recommendation-system"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f34b4f1a",
   "metadata": {
    "papermill": {
     "duration": 0.014144,
     "end_time": "2023-01-07T07:06:50.995303",
     "exception": false,
     "start_time": "2023-01-07T07:06:50.981159",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Importing Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "6669dec8",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:51.025923Z",
     "iopub.status.busy": "2023-01-07T07:06:51.025302Z",
     "iopub.status.idle": "2023-01-07T07:06:52.644614Z",
     "shell.execute_reply": "2023-01-07T07:06:52.643301Z"
    },
    "papermill": {
     "duration": 1.640498,
     "end_time": "2023-01-07T07:06:52.648151",
     "exception": false,
     "start_time": "2023-01-07T07:06:51.007653",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# EDA\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "# Data Preprocessing \n",
    "from sklearn import preprocessing\n",
    "\n",
    "# Data visualisation\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "sns.set()\n",
    "\n",
    "# Recommender System Imps\n",
    "# Content Based Filtering \n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.metrics.pairwise import linear_kernel\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "# Collaborative Based Filtering \n",
    "from scipy.sparse import csr_matrix\n",
    "from sklearn.neighbors import NearestNeighbors\n",
    "\n",
    "# To work with text data \n",
    "import re\n",
    "import string"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "444c9a00",
   "metadata": {
    "papermill": {
     "duration": 0.012295,
     "end_time": "2023-01-07T07:06:52.673173",
     "exception": false,
     "start_time": "2023-01-07T07:06:52.660878",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Importing Data "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "93fcb308",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:52.700331Z",
     "iopub.status.busy": "2023-01-07T07:06:52.699908Z",
     "iopub.status.idle": "2023-01-07T07:06:52.740094Z",
     "shell.execute_reply": "2023-01-07T07:06:52.739055Z"
    },
    "papermill": {
     "duration": 0.057215,
     "end_time": "2023-01-07T07:06:52.743037",
     "exception": false,
     "start_time": "2023-01-07T07:06:52.685822",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Food_ID</th>\n",
       "      <th>Name</th>\n",
       "      <th>C_Type</th>\n",
       "      <th>Veg_Non</th>\n",
       "      <th>Describe</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>summer squash salad</td>\n",
       "      <td>Healthy Food</td>\n",
       "      <td>veg</td>\n",
       "      <td>white balsamic vinegar, lemon juice, lemon rin...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>chicken minced salad</td>\n",
       "      <td>Healthy Food</td>\n",
       "      <td>non-veg</td>\n",
       "      <td>olive oil, chicken mince, garlic (minced), oni...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>sweet chilli almonds</td>\n",
       "      <td>Snack</td>\n",
       "      <td>veg</td>\n",
       "      <td>almonds whole, egg white, curry leaves, salt, ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>tricolour salad</td>\n",
       "      <td>Healthy Food</td>\n",
       "      <td>veg</td>\n",
       "      <td>vinegar, honey/sugar, soy sauce, salt, garlic ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>christmas cake</td>\n",
       "      <td>Dessert</td>\n",
       "      <td>veg</td>\n",
       "      <td>christmas dry fruits (pre-soaked), orange zest...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Food_ID                  Name        C_Type  Veg_Non  \\\n",
       "0        1   summer squash salad  Healthy Food      veg   \n",
       "1        2  chicken minced salad  Healthy Food  non-veg   \n",
       "2        3  sweet chilli almonds         Snack      veg   \n",
       "3        4       tricolour salad  Healthy Food      veg   \n",
       "4        5        christmas cake       Dessert      veg   \n",
       "\n",
       "                                            Describe  \n",
       "0  white balsamic vinegar, lemon juice, lemon rin...  \n",
       "1  olive oil, chicken mince, garlic (minced), oni...  \n",
       "2  almonds whole, egg white, curry leaves, salt, ...  \n",
       "3  vinegar, honey/sugar, soy sauce, salt, garlic ...  \n",
       "4  christmas dry fruits (pre-soaked), orange zest...  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv('/kaggle/input/food-recommendation-system/1662574418893344.csv')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "2589be4b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:52.771452Z",
     "iopub.status.busy": "2023-01-07T07:06:52.770666Z",
     "iopub.status.idle": "2023-01-07T07:06:52.785240Z",
     "shell.execute_reply": "2023-01-07T07:06:52.783768Z"
    },
    "papermill": {
     "duration": 0.031842,
     "end_time": "2023-01-07T07:06:52.787870",
     "exception": false,
     "start_time": "2023-01-07T07:06:52.756028",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "400"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# No of dishes in my dataset\n",
    "len(list(df['Name'].unique()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "91f49606",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:52.817724Z",
     "iopub.status.busy": "2023-01-07T07:06:52.817281Z",
     "iopub.status.idle": "2023-01-07T07:06:52.829086Z",
     "shell.execute_reply": "2023-01-07T07:06:52.827488Z"
    },
    "papermill": {
     "duration": 0.030315,
     "end_time": "2023-01-07T07:06:52.832440",
     "exception": false,
     "start_time": "2023-01-07T07:06:52.802125",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Healthy Food', 'Snack', 'Dessert', 'Japanese', 'Indian', 'French',\n",
       "       'Mexican', 'Italian', 'Chinese', 'Beverage', 'Thai', 'Korean',\n",
       "       ' Korean', 'Vietnames', 'Nepalese', 'Spanish'], dtype=object)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['C_Type'].unique() # Categorical Data "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "8ca60789",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:52.875844Z",
     "iopub.status.busy": "2023-01-07T07:06:52.872372Z",
     "iopub.status.idle": "2023-01-07T07:06:52.885641Z",
     "shell.execute_reply": "2023-01-07T07:06:52.884076Z"
    },
    "papermill": {
     "duration": 0.037762,
     "end_time": "2023-01-07T07:06:52.889010",
     "exception": false,
     "start_time": "2023-01-07T07:06:52.851248",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['veg', 'non-veg'], dtype=object)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Veg_Non'].unique() # Categorical Data "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "c41ea3b4",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:52.926995Z",
     "iopub.status.busy": "2023-01-07T07:06:52.926018Z",
     "iopub.status.idle": "2023-01-07T07:06:52.934005Z",
     "shell.execute_reply": "2023-01-07T07:06:52.932956Z"
    },
    "papermill": {
     "duration": 0.030997,
     "end_time": "2023-01-07T07:06:52.936396",
     "exception": false,
     "start_time": "2023-01-07T07:06:52.905399",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "400"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(df) \n",
    "# Hmm... Small Dataset!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "fce87629",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:52.968106Z",
     "iopub.status.busy": "2023-01-07T07:06:52.967123Z",
     "iopub.status.idle": "2023-01-07T07:06:52.975358Z",
     "shell.execute_reply": "2023-01-07T07:06:52.974207Z"
    },
    "papermill": {
     "duration": 0.026908,
     "end_time": "2023-01-07T07:06:52.978099",
     "exception": false,
     "start_time": "2023-01-07T07:06:52.951191",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(400, 5)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "1044d97a",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:53.007720Z",
     "iopub.status.busy": "2023-01-07T07:06:53.007229Z",
     "iopub.status.idle": "2023-01-07T07:06:53.030983Z",
     "shell.execute_reply": "2023-01-07T07:06:53.028945Z"
    },
    "papermill": {
     "duration": 0.042866,
     "end_time": "2023-01-07T07:06:53.035192",
     "exception": false,
     "start_time": "2023-01-07T07:06:52.992326",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 400 entries, 0 to 399\n",
      "Data columns (total 5 columns):\n",
      " #   Column    Non-Null Count  Dtype \n",
      "---  ------    --------------  ----- \n",
      " 0   Food_ID   400 non-null    int64 \n",
      " 1   Name      400 non-null    object\n",
      " 2   C_Type    400 non-null    object\n",
      " 3   Veg_Non   400 non-null    object\n",
      " 4   Describe  400 non-null    object\n",
      "dtypes: int64(1), object(4)\n",
      "memory usage: 15.8+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "a85f3027",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:53.070348Z",
     "iopub.status.busy": "2023-01-07T07:06:53.068987Z",
     "iopub.status.idle": "2023-01-07T07:06:53.077345Z",
     "shell.execute_reply": "2023-01-07T07:06:53.075743Z"
    },
    "papermill": {
     "duration": 0.029854,
     "end_time": "2023-01-07T07:06:53.080838",
     "exception": false,
     "start_time": "2023-01-07T07:06:53.050984",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Let's make a function to remove all the punctuation from the \"Describe\" column\n",
    "def text_cleaning(text):\n",
    "    text  = \"\".join([char for char in text if char not in string.punctuation])    \n",
    "    return text"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "3b97e69d",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:53.114947Z",
     "iopub.status.busy": "2023-01-07T07:06:53.114023Z",
     "iopub.status.idle": "2023-01-07T07:06:53.129659Z",
     "shell.execute_reply": "2023-01-07T07:06:53.128231Z"
    },
    "papermill": {
     "duration": 0.035952,
     "end_time": "2023-01-07T07:06:53.132203",
     "exception": false,
     "start_time": "2023-01-07T07:06:53.096251",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Let's clean the text \n",
    "df['Describe'] = df['Describe'].apply(text_cleaning)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "805ac839",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:53.162013Z",
     "iopub.status.busy": "2023-01-07T07:06:53.161318Z",
     "iopub.status.idle": "2023-01-07T07:06:53.174382Z",
     "shell.execute_reply": "2023-01-07T07:06:53.172731Z"
    },
    "papermill": {
     "duration": 0.031619,
     "end_time": "2023-01-07T07:06:53.177505",
     "exception": false,
     "start_time": "2023-01-07T07:06:53.145886",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Food_ID</th>\n",
       "      <th>Name</th>\n",
       "      <th>C_Type</th>\n",
       "      <th>Veg_Non</th>\n",
       "      <th>Describe</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>summer squash salad</td>\n",
       "      <td>Healthy Food</td>\n",
       "      <td>veg</td>\n",
       "      <td>white balsamic vinegar lemon juice lemon rind ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>chicken minced salad</td>\n",
       "      <td>Healthy Food</td>\n",
       "      <td>non-veg</td>\n",
       "      <td>olive oil chicken mince garlic minced onion sa...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>sweet chilli almonds</td>\n",
       "      <td>Snack</td>\n",
       "      <td>veg</td>\n",
       "      <td>almonds whole egg white curry leaves salt suga...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>tricolour salad</td>\n",
       "      <td>Healthy Food</td>\n",
       "      <td>veg</td>\n",
       "      <td>vinegar honeysugar soy sauce salt garlic clove...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>christmas cake</td>\n",
       "      <td>Dessert</td>\n",
       "      <td>veg</td>\n",
       "      <td>christmas dry fruits presoaked orange zest lem...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Food_ID                  Name        C_Type  Veg_Non  \\\n",
       "0        1   summer squash salad  Healthy Food      veg   \n",
       "1        2  chicken minced salad  Healthy Food  non-veg   \n",
       "2        3  sweet chilli almonds         Snack      veg   \n",
       "3        4       tricolour salad  Healthy Food      veg   \n",
       "4        5        christmas cake       Dessert      veg   \n",
       "\n",
       "                                            Describe  \n",
       "0  white balsamic vinegar lemon juice lemon rind ...  \n",
       "1  olive oil chicken mince garlic minced onion sa...  \n",
       "2  almonds whole egg white curry leaves salt suga...  \n",
       "3  vinegar honeysugar soy sauce salt garlic clove...  \n",
       "4  christmas dry fruits presoaked orange zest lem...  "
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Let's see if that worked...\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "15bf69b6",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:53.209212Z",
     "iopub.status.busy": "2023-01-07T07:06:53.208794Z",
     "iopub.status.idle": "2023-01-07T07:06:53.219325Z",
     "shell.execute_reply": "2023-01-07T07:06:53.218343Z"
    },
    "papermill": {
     "duration": 0.029981,
     "end_time": "2023-01-07T07:06:53.221820",
     "exception": false,
     "start_time": "2023-01-07T07:06:53.191839",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Are there any duplicate data ?\n",
    "df.duplicated().sum()\n",
    "# None :)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "99b9dd70",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:53.252566Z",
     "iopub.status.busy": "2023-01-07T07:06:53.251182Z",
     "iopub.status.idle": "2023-01-07T07:06:53.261294Z",
     "shell.execute_reply": "2023-01-07T07:06:53.260380Z"
    },
    "papermill": {
     "duration": 0.027498,
     "end_time": "2023-01-07T07:06:53.263497",
     "exception": false,
     "start_time": "2023-01-07T07:06:53.235999",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Food_ID     0\n",
       "Name        0\n",
       "C_Type      0\n",
       "Veg_Non     0\n",
       "Describe    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Are there any null values?\n",
    "df.isnull().sum()\n",
    "# None :))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "ac0a221f",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:53.297548Z",
     "iopub.status.busy": "2023-01-07T07:06:53.296840Z",
     "iopub.status.idle": "2023-01-07T07:06:53.315512Z",
     "shell.execute_reply": "2023-01-07T07:06:53.313875Z"
    },
    "papermill": {
     "duration": 0.03981,
     "end_time": "2023-01-07T07:06:53.318536",
     "exception": false,
     "start_time": "2023-01-07T07:06:53.278726",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Food_ID</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>400.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>200.500000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>115.614301</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>100.750000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>200.500000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>300.250000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>400.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          Food_ID\n",
       "count  400.000000\n",
       "mean   200.500000\n",
       "std    115.614301\n",
       "min      1.000000\n",
       "25%    100.750000\n",
       "50%    200.500000\n",
       "75%    300.250000\n",
       "max    400.000000"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# General Description \n",
    "df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b9845d97",
   "metadata": {
    "papermill": {
     "duration": 0.013934,
     "end_time": "2023-01-07T07:06:53.347066",
     "exception": false,
     "start_time": "2023-01-07T07:06:53.333132",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Content Based Filtering "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "930dc724",
   "metadata": {
    "papermill": {
     "duration": 0.013749,
     "end_time": "2023-01-07T07:06:53.374817",
     "exception": false,
     "start_time": "2023-01-07T07:06:53.361068",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# 1. Simple Content Based Filtering"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "463f1113",
   "metadata": {
    "papermill": {
     "duration": 0.013594,
     "end_time": "2023-01-07T07:06:53.402441",
     "exception": false,
     "start_time": "2023-01-07T07:06:53.388847",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### What is TF-IDF?\n",
    "In a large text corpus, some words will be very present (e.g. “the”, “a”, “is” in English) hence carrying very little meaningful information about the actual contents of the document. If we were to feed the direct count data directly to a classifier those very frequent terms would shadow the frequencies of rarer yet more interesting terms.\n",
    "\n",
    "In order to re-weight the count features into floating point values suitable for usage by a classifier it is very common to use the tf–idf transform.\n",
    "\n",
    "Thanks to https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "82cc2712",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:53.433297Z",
     "iopub.status.busy": "2023-01-07T07:06:53.432148Z",
     "iopub.status.idle": "2023-01-07T07:06:53.462355Z",
     "shell.execute_reply": "2023-01-07T07:06:53.461104Z"
    },
    "papermill": {
     "duration": 0.048596,
     "end_time": "2023-01-07T07:06:53.464950",
     "exception": false,
     "start_time": "2023-01-07T07:06:53.416354",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(400, 1261)"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tfidf = TfidfVectorizer(stop_words='english')\n",
    "tfidf_matrix = tfidf.fit_transform(df['Describe'])\n",
    "tfidf_matrix.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9707dd9d",
   "metadata": {
    "papermill": {
     "duration": 0.013795,
     "end_time": "2023-01-07T07:06:53.492976",
     "exception": false,
     "start_time": "2023-01-07T07:06:53.479181",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### What is Linear Kernel ?\n",
    "The function linear_kernel computes the linear kernel, that is, a special case of polynomial_kernel with degree=1 and coef0=0 (homogeneous). \n",
    "\n",
    "Thanks to https://scikit-learn.org/stable/modules/metrics.html#linear-kernel"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "75fa67fb",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:53.523266Z",
     "iopub.status.busy": "2023-01-07T07:06:53.522430Z",
     "iopub.status.idle": "2023-01-07T07:06:53.537793Z",
     "shell.execute_reply": "2023-01-07T07:06:53.536467Z"
    },
    "papermill": {
     "duration": 0.034186,
     "end_time": "2023-01-07T07:06:53.541270",
     "exception": false,
     "start_time": "2023-01-07T07:06:53.507084",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1.        , 0.16228366, 0.13001124, ..., 0.1286286 , 0.04277223,\n",
       "        0.09993639],\n",
       "       [0.16228366, 1.        , 0.06799336, ..., 0.14878001, 0.05688681,\n",
       "        0.16917639],\n",
       "       [0.13001124, 0.06799336, 1.        , ..., 0.03291577, 0.11795401,\n",
       "        0.01834168],\n",
       "       ...,\n",
       "       [0.1286286 , 0.14878001, 0.03291577, ..., 1.        , 0.        ,\n",
       "        0.10087579],\n",
       "       [0.04277223, 0.05688681, 0.11795401, ..., 0.        , 1.        ,\n",
       "        0.        ],\n",
       "       [0.09993639, 0.16917639, 0.01834168, ..., 0.10087579, 0.        ,\n",
       "        1.        ]])"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)\n",
    "cosine_sim"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "6a366eb7",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:53.572862Z",
     "iopub.status.busy": "2023-01-07T07:06:53.572402Z",
     "iopub.status.idle": "2023-01-07T07:06:53.583259Z",
     "shell.execute_reply": "2023-01-07T07:06:53.582070Z"
    },
    "papermill": {
     "duration": 0.029876,
     "end_time": "2023-01-07T07:06:53.585993",
     "exception": false,
     "start_time": "2023-01-07T07:06:53.556117",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Name\n",
       "summer squash salad                                          0\n",
       "chicken minced salad                                         1\n",
       "sweet chilli almonds                                         2\n",
       "tricolour salad                                              3\n",
       "christmas cake                                               4\n",
       "                                                          ... \n",
       "Kimchi Toast                                               395\n",
       "Tacos de Gobernador (Shrimp, Poblano, and Cheese Tacos)    396\n",
       "Melted Broccoli Pasta With Capers and Anchovies            397\n",
       "Lemon-Ginger Cake with Pistachios                          398\n",
       "Rosemary Roasted Vegetables                                399\n",
       "Length: 400, dtype: int64"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Just considering the Food names from the dataframe\n",
    "indices = pd.Series(df.index, index=df['Name']).drop_duplicates()\n",
    "indices"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "51ccb29b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:53.618017Z",
     "iopub.status.busy": "2023-01-07T07:06:53.617579Z",
     "iopub.status.idle": "2023-01-07T07:06:53.625005Z",
     "shell.execute_reply": "2023-01-07T07:06:53.623742Z"
    },
    "papermill": {
     "duration": 0.026047,
     "end_time": "2023-01-07T07:06:53.627384",
     "exception": false,
     "start_time": "2023-01-07T07:06:53.601337",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# The main recommender code!\n",
    "def get_recommendations(title, cosine_sim=cosine_sim):\n",
    "    \n",
    "    idx = indices[title]\n",
    "    sim_scores = list(enumerate(cosine_sim[idx]))\n",
    "    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)\n",
    "\n",
    "    # Get the scores of the 5 most similar food\n",
    "    sim_scores = sim_scores[1:6]\n",
    "    \n",
    "    food_indices = [i[0] for i in sim_scores]\n",
    "    return df['Name'].iloc[food_indices]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0db228ca",
   "metadata": {
    "papermill": {
     "duration": 0.014902,
     "end_time": "2023-01-07T07:06:53.656714",
     "exception": false,
     "start_time": "2023-01-07T07:06:53.641812",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# 2. Advanced Content Based Filtering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "842facbb",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:53.690696Z",
     "iopub.status.busy": "2023-01-07T07:06:53.689756Z",
     "iopub.status.idle": "2023-01-07T07:06:53.695720Z",
     "shell.execute_reply": "2023-01-07T07:06:53.694215Z"
    },
    "papermill": {
     "duration": 0.026191,
     "end_time": "2023-01-07T07:06:53.698698",
     "exception": false,
     "start_time": "2023-01-07T07:06:53.672507",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Including all features that will help in recommending better\n",
    "features = ['C_Type','Veg_Non', 'Describe']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "dc585e1f",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:53.730380Z",
     "iopub.status.busy": "2023-01-07T07:06:53.729529Z",
     "iopub.status.idle": "2023-01-07T07:06:53.735554Z",
     "shell.execute_reply": "2023-01-07T07:06:53.734208Z"
    },
    "papermill": {
     "duration": 0.024705,
     "end_time": "2023-01-07T07:06:53.738373",
     "exception": false,
     "start_time": "2023-01-07T07:06:53.713668",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Soup represents a mixture of elements \n",
    "# Similarly, I am making one column that will have all the important features \n",
    "# I am simply concatenating the strings \n",
    "\n",
    "def create_soup(x):\n",
    "    return x['C_Type'] + \" \" + x['Veg_Non'] + \" \" + x['Describe']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "9e41d321",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:53.770581Z",
     "iopub.status.busy": "2023-01-07T07:06:53.770165Z",
     "iopub.status.idle": "2023-01-07T07:06:53.785485Z",
     "shell.execute_reply": "2023-01-07T07:06:53.783901Z"
    },
    "papermill": {
     "duration": 0.035238,
     "end_time": "2023-01-07T07:06:53.788369",
     "exception": false,
     "start_time": "2023-01-07T07:06:53.753131",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Using the soup(), I am creating the column for the dataframe df\n",
    "df['soup'] = df.apply(create_soup, axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "0d16ae8a",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:53.820854Z",
     "iopub.status.busy": "2023-01-07T07:06:53.820393Z",
     "iopub.status.idle": "2023-01-07T07:06:53.835026Z",
     "shell.execute_reply": "2023-01-07T07:06:53.833737Z"
    },
    "papermill": {
     "duration": 0.033891,
     "end_time": "2023-01-07T07:06:53.837736",
     "exception": false,
     "start_time": "2023-01-07T07:06:53.803845",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Food_ID</th>\n",
       "      <th>Name</th>\n",
       "      <th>C_Type</th>\n",
       "      <th>Veg_Non</th>\n",
       "      <th>Describe</th>\n",
       "      <th>soup</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>summer squash salad</td>\n",
       "      <td>Healthy Food</td>\n",
       "      <td>veg</td>\n",
       "      <td>white balsamic vinegar lemon juice lemon rind ...</td>\n",
       "      <td>Healthy Food veg white balsamic vinegar lemon ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>chicken minced salad</td>\n",
       "      <td>Healthy Food</td>\n",
       "      <td>non-veg</td>\n",
       "      <td>olive oil chicken mince garlic minced onion sa...</td>\n",
       "      <td>Healthy Food non-veg olive oil chicken mince g...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>sweet chilli almonds</td>\n",
       "      <td>Snack</td>\n",
       "      <td>veg</td>\n",
       "      <td>almonds whole egg white curry leaves salt suga...</td>\n",
       "      <td>Snack veg almonds whole egg white curry leaves...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>tricolour salad</td>\n",
       "      <td>Healthy Food</td>\n",
       "      <td>veg</td>\n",
       "      <td>vinegar honeysugar soy sauce salt garlic clove...</td>\n",
       "      <td>Healthy Food veg vinegar honeysugar soy sauce ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>christmas cake</td>\n",
       "      <td>Dessert</td>\n",
       "      <td>veg</td>\n",
       "      <td>christmas dry fruits presoaked orange zest lem...</td>\n",
       "      <td>Dessert veg christmas dry fruits presoaked ora...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Food_ID                  Name        C_Type  Veg_Non  \\\n",
       "0        1   summer squash salad  Healthy Food      veg   \n",
       "1        2  chicken minced salad  Healthy Food  non-veg   \n",
       "2        3  sweet chilli almonds         Snack      veg   \n",
       "3        4       tricolour salad  Healthy Food      veg   \n",
       "4        5        christmas cake       Dessert      veg   \n",
       "\n",
       "                                            Describe  \\\n",
       "0  white balsamic vinegar lemon juice lemon rind ...   \n",
       "1  olive oil chicken mince garlic minced onion sa...   \n",
       "2  almonds whole egg white curry leaves salt suga...   \n",
       "3  vinegar honeysugar soy sauce salt garlic clove...   \n",
       "4  christmas dry fruits presoaked orange zest lem...   \n",
       "\n",
       "                                                soup  \n",
       "0  Healthy Food veg white balsamic vinegar lemon ...  \n",
       "1  Healthy Food non-veg olive oil chicken mince g...  \n",
       "2  Snack veg almonds whole egg white curry leaves...  \n",
       "3  Healthy Food veg vinegar honeysugar soy sauce ...  \n",
       "4  Dessert veg christmas dry fruits presoaked ora...  "
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Checking out if that worked!\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "74896423",
   "metadata": {
    "papermill": {
     "duration": 0.015647,
     "end_time": "2023-01-07T07:06:53.868928",
     "exception": false,
     "start_time": "2023-01-07T07:06:53.853281",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### What is Count Vectorizer ?\n",
    "Convert a collection of text documents to a matrix of token counts. This implementation produces a sparse representation of the counts using scipy.sparse.csr_matrix.\n",
    "\n",
    "### What is fit_transform ?\n",
    "Learn the vocabulary dictionary and return document-term matrix.\n",
    "\n",
    "Thanks to https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "2c71b3b6",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:53.902702Z",
     "iopub.status.busy": "2023-01-07T07:06:53.901589Z",
     "iopub.status.idle": "2023-01-07T07:06:53.920553Z",
     "shell.execute_reply": "2023-01-07T07:06:53.919627Z"
    },
    "papermill": {
     "duration": 0.039144,
     "end_time": "2023-01-07T07:06:53.923126",
     "exception": false,
     "start_time": "2023-01-07T07:06:53.883982",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "count = CountVectorizer(stop_words='english')\n",
    "count_matrix = count.fit_transform(df['soup'])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "68d02810",
   "metadata": {
    "papermill": {
     "duration": 0.014318,
     "end_time": "2023-01-07T07:06:53.952221",
     "exception": false,
     "start_time": "2023-01-07T07:06:53.937903",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### What is Cosine Similarity? \n",
    "The function computes cosine similarity between samples in X and Y.\n",
    "\n",
    "Thanks to https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html\n",
    "\n",
    "cosine_similarity computes the L2-normalized dot product of vectors. This is called cosine similarity, because Euclidean (L2) normalization projects the vectors onto the unit sphere, and their dot product is then the cosine of the angle between the points denoted by the vectors.\n",
    "\n",
    "Thanks to https://scikit-learn.org/stable/modules/metrics.html#cosine-similarity"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "4828ae10",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:53.985525Z",
     "iopub.status.busy": "2023-01-07T07:06:53.985099Z",
     "iopub.status.idle": "2023-01-07T07:06:54.001767Z",
     "shell.execute_reply": "2023-01-07T07:06:54.000638Z"
    },
    "papermill": {
     "duration": 0.036367,
     "end_time": "2023-01-07T07:06:54.004468",
     "exception": false,
     "start_time": "2023-01-07T07:06:53.968101",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "cosine_sim2 = cosine_similarity(count_matrix, count_matrix)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "654ae667",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:54.036447Z",
     "iopub.status.busy": "2023-01-07T07:06:54.036044Z",
     "iopub.status.idle": "2023-01-07T07:06:54.043571Z",
     "shell.execute_reply": "2023-01-07T07:06:54.042461Z"
    },
    "papermill": {
     "duration": 0.026958,
     "end_time": "2023-01-07T07:06:54.046482",
     "exception": false,
     "start_time": "2023-01-07T07:06:54.019524",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Reseting the index and pulling out the names of the food alone from the df dataframe\n",
    "df = df.reset_index()\n",
    "indices = pd.Series(df.index, index=df['Name'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "1616ef4b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:54.080487Z",
     "iopub.status.busy": "2023-01-07T07:06:54.080058Z",
     "iopub.status.idle": "2023-01-07T07:06:54.089659Z",
     "shell.execute_reply": "2023-01-07T07:06:54.088435Z"
    },
    "papermill": {
     "duration": 0.029999,
     "end_time": "2023-01-07T07:06:54.092013",
     "exception": false,
     "start_time": "2023-01-07T07:06:54.062014",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Name\n",
       "summer squash salad                                          0\n",
       "chicken minced salad                                         1\n",
       "sweet chilli almonds                                         2\n",
       "tricolour salad                                              3\n",
       "christmas cake                                               4\n",
       "                                                          ... \n",
       "Kimchi Toast                                               395\n",
       "Tacos de Gobernador (Shrimp, Poblano, and Cheese Tacos)    396\n",
       "Melted Broccoli Pasta With Capers and Anchovies            397\n",
       "Lemon-Ginger Cake with Pistachios                          398\n",
       "Rosemary Roasted Vegetables                                399\n",
       "Length: 400, dtype: int64"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Let's see the names of the food pulled out\n",
    "display(indices)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dcf8194a",
   "metadata": {
    "papermill": {
     "duration": 0.014704,
     "end_time": "2023-01-07T07:06:54.121421",
     "exception": false,
     "start_time": "2023-01-07T07:06:54.106717",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Testing Content Based Filtering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "3c788431",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:54.153943Z",
     "iopub.status.busy": "2023-01-07T07:06:54.152811Z",
     "iopub.status.idle": "2023-01-07T07:06:54.162208Z",
     "shell.execute_reply": "2023-01-07T07:06:54.161206Z"
    },
    "papermill": {
     "duration": 0.028366,
     "end_time": "2023-01-07T07:06:54.164401",
     "exception": false,
     "start_time": "2023-01-07T07:06:54.136035",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "103             chilli chicken\n",
       "1         chicken minced salad\n",
       "27     vegetable som tam salad\n",
       "282          veg hakka noodles\n",
       "166             veg fried rice\n",
       "Name: Name, dtype: object"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# This is the first model - simple variation \n",
    "get_recommendations('tricolour salad')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "4d263379",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:54.197925Z",
     "iopub.status.busy": "2023-01-07T07:06:54.197463Z",
     "iopub.status.idle": "2023-01-07T07:06:54.207188Z",
     "shell.execute_reply": "2023-01-07T07:06:54.205657Z"
    },
    "papermill": {
     "duration": 0.028861,
     "end_time": "2023-01-07T07:06:54.209715",
     "exception": false,
     "start_time": "2023-01-07T07:06:54.180854",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1                         chicken minced salad\n",
       "103                             chilli chicken\n",
       "27                     vegetable som tam salad\n",
       "177                        oats shallots pulao\n",
       "69     shepherds salad (tamatar-kheera salaad)\n",
       "Name: Name, dtype: object"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# This is the second model - advanced variation \n",
    "get_recommendations('tricolour salad', cosine_sim2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "60777d3b",
   "metadata": {
    "papermill": {
     "duration": 0.015787,
     "end_time": "2023-01-07T07:06:54.240304",
     "exception": false,
     "start_time": "2023-01-07T07:06:54.224517",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Collaborative Filtering "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "94652695",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:54.272597Z",
     "iopub.status.busy": "2023-01-07T07:06:54.272124Z",
     "iopub.status.idle": "2023-01-07T07:06:54.291313Z",
     "shell.execute_reply": "2023-01-07T07:06:54.289290Z"
    },
    "papermill": {
     "duration": 0.039289,
     "end_time": "2023-01-07T07:06:54.294539",
     "exception": false,
     "start_time": "2023-01-07T07:06:54.255250",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>User_ID</th>\n",
       "      <th>Food_ID</th>\n",
       "      <th>Rating</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1.0</td>\n",
       "      <td>88.0</td>\n",
       "      <td>4.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1.0</td>\n",
       "      <td>46.0</td>\n",
       "      <td>3.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1.0</td>\n",
       "      <td>24.0</td>\n",
       "      <td>5.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1.0</td>\n",
       "      <td>25.0</td>\n",
       "      <td>4.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2.0</td>\n",
       "      <td>49.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   User_ID  Food_ID  Rating\n",
       "0      1.0     88.0     4.0\n",
       "1      1.0     46.0     3.0\n",
       "2      1.0     24.0     5.0\n",
       "3      1.0     25.0     4.0\n",
       "4      2.0     49.0     1.0"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Importing the ratings file\n",
    "rating = pd.read_csv('/kaggle/input/food-recommendation-system/ratings.csv')\n",
    "rating.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "d1b580de",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:54.328750Z",
     "iopub.status.busy": "2023-01-07T07:06:54.328286Z",
     "iopub.status.idle": "2023-01-07T07:06:54.335588Z",
     "shell.execute_reply": "2023-01-07T07:06:54.334191Z"
    },
    "papermill": {
     "duration": 0.028153,
     "end_time": "2023-01-07T07:06:54.338743",
     "exception": false,
     "start_time": "2023-01-07T07:06:54.310590",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(512, 3)"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Checking the shape\n",
    "rating.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "a2a96399",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:54.373145Z",
     "iopub.status.busy": "2023-01-07T07:06:54.372668Z",
     "iopub.status.idle": "2023-01-07T07:06:54.382342Z",
     "shell.execute_reply": "2023-01-07T07:06:54.381137Z"
    },
    "papermill": {
     "duration": 0.02987,
     "end_time": "2023-01-07T07:06:54.384841",
     "exception": false,
     "start_time": "2023-01-07T07:06:54.354971",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "User_ID    1\n",
       "Food_ID    1\n",
       "Rating     1\n",
       "dtype: int64"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Checking for null values \n",
    "rating.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "6b9e1c5c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:54.418160Z",
     "iopub.status.busy": "2023-01-07T07:06:54.417744Z",
     "iopub.status.idle": "2023-01-07T07:06:54.431717Z",
     "shell.execute_reply": "2023-01-07T07:06:54.430456Z"
    },
    "papermill": {
     "duration": 0.033442,
     "end_time": "2023-01-07T07:06:54.434242",
     "exception": false,
     "start_time": "2023-01-07T07:06:54.400800",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>User_ID</th>\n",
       "      <th>Food_ID</th>\n",
       "      <th>Rating</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>507</th>\n",
       "      <td>99.0</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>508</th>\n",
       "      <td>100.0</td>\n",
       "      <td>24.0</td>\n",
       "      <td>10.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>509</th>\n",
       "      <td>100.0</td>\n",
       "      <td>233.0</td>\n",
       "      <td>10.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>510</th>\n",
       "      <td>100.0</td>\n",
       "      <td>29.0</td>\n",
       "      <td>7.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>511</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     User_ID  Food_ID  Rating\n",
       "507     99.0     22.0     1.0\n",
       "508    100.0     24.0    10.0\n",
       "509    100.0    233.0    10.0\n",
       "510    100.0     29.0     7.0\n",
       "511      NaN      NaN     NaN"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# I actually saw the data earlier and found that the last row had no values \n",
    "# Let's see \n",
    "rating.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "afc165b8",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:54.468905Z",
     "iopub.status.busy": "2023-01-07T07:06:54.468488Z",
     "iopub.status.idle": "2023-01-07T07:06:54.481474Z",
     "shell.execute_reply": "2023-01-07T07:06:54.480181Z"
    },
    "papermill": {
     "duration": 0.033684,
     "end_time": "2023-01-07T07:06:54.484126",
     "exception": false,
     "start_time": "2023-01-07T07:06:54.450442",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>User_ID</th>\n",
       "      <th>Food_ID</th>\n",
       "      <th>Rating</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>506</th>\n",
       "      <td>99.0</td>\n",
       "      <td>65.0</td>\n",
       "      <td>7.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>507</th>\n",
       "      <td>99.0</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>508</th>\n",
       "      <td>100.0</td>\n",
       "      <td>24.0</td>\n",
       "      <td>10.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>509</th>\n",
       "      <td>100.0</td>\n",
       "      <td>233.0</td>\n",
       "      <td>10.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>510</th>\n",
       "      <td>100.0</td>\n",
       "      <td>29.0</td>\n",
       "      <td>7.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     User_ID  Food_ID  Rating\n",
       "506     99.0     65.0     7.0\n",
       "507     99.0     22.0     1.0\n",
       "508    100.0     24.0    10.0\n",
       "509    100.0    233.0    10.0\n",
       "510    100.0     29.0     7.0"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Removing the last row \n",
    "rating = rating[:511]\n",
    "rating.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "b62be650",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:54.518328Z",
     "iopub.status.busy": "2023-01-07T07:06:54.517895Z",
     "iopub.status.idle": "2023-01-07T07:06:54.526301Z",
     "shell.execute_reply": "2023-01-07T07:06:54.525293Z"
    },
    "papermill": {
     "duration": 0.028582,
     "end_time": "2023-01-07T07:06:54.528624",
     "exception": false,
     "start_time": "2023-01-07T07:06:54.500042",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "User_ID    0\n",
       "Food_ID    0\n",
       "Rating     0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# So, now there should not be any null value \n",
    "rating.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "c907c04b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:54.566445Z",
     "iopub.status.busy": "2023-01-07T07:06:54.566007Z",
     "iopub.status.idle": "2023-01-07T07:06:54.588007Z",
     "shell.execute_reply": "2023-01-07T07:06:54.586561Z"
    },
    "papermill": {
     "duration": 0.04505,
     "end_time": "2023-01-07T07:06:54.591160",
     "exception": false,
     "start_time": "2023-01-07T07:06:54.546110",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Food_ID</th>\n",
       "      <th>Rating_count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1.0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2.0</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3.0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4.0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5.0</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>304</th>\n",
       "      <td>305.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>305</th>\n",
       "      <td>306.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>306</th>\n",
       "      <td>307.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>307</th>\n",
       "      <td>308.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>308</th>\n",
       "      <td>309.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>309 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Food_ID  Rating_count\n",
       "0        1.0             2\n",
       "1        2.0             3\n",
       "2        3.0             2\n",
       "3        4.0             2\n",
       "4        5.0             6\n",
       "..       ...           ...\n",
       "304    305.0             1\n",
       "305    306.0             1\n",
       "306    307.0             1\n",
       "307    308.0             1\n",
       "308    309.0             1\n",
       "\n",
       "[309 rows x 2 columns]"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Making a dataframe that has food ID and the number of ratings\n",
    "food_rating = rating.groupby(by = 'Food_ID').count()\n",
    "food_rating = food_rating['Rating'].reset_index().rename(columns={'Rating':'Rating_count'})\n",
    "food_rating"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "d93fc314",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:54.625392Z",
     "iopub.status.busy": "2023-01-07T07:06:54.624950Z",
     "iopub.status.idle": "2023-01-07T07:06:54.637226Z",
     "shell.execute_reply": "2023-01-07T07:06:54.635956Z"
    },
    "papermill": {
     "duration": 0.032538,
     "end_time": "2023-01-07T07:06:54.640110",
     "exception": false,
     "start_time": "2023-01-07T07:06:54.607572",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count    309.000000\n",
       "mean       1.653722\n",
       "std        1.107748\n",
       "min        1.000000\n",
       "25%        1.000000\n",
       "50%        1.000000\n",
       "75%        2.000000\n",
       "max        7.000000\n",
       "Name: Rating_count, dtype: float64"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Food rating dataframe description\n",
    "food_rating['Rating_count'].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "0957a642",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:54.675504Z",
     "iopub.status.busy": "2023-01-07T07:06:54.675093Z",
     "iopub.status.idle": "2023-01-07T07:06:54.696293Z",
     "shell.execute_reply": "2023-01-07T07:06:54.695124Z"
    },
    "papermill": {
     "duration": 0.041776,
     "end_time": "2023-01-07T07:06:54.699001",
     "exception": false,
     "start_time": "2023-01-07T07:06:54.657225",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>User_ID</th>\n",
       "      <th>Rating_count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1.0</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2.0</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3.0</td>\n",
       "      <td>9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4.0</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5.0</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>95</th>\n",
       "      <td>96.0</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>96</th>\n",
       "      <td>97.0</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>97</th>\n",
       "      <td>98.0</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>98</th>\n",
       "      <td>99.0</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>99</th>\n",
       "      <td>100.0</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>100 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "    User_ID  Rating_count\n",
       "0       1.0             4\n",
       "1       2.0             4\n",
       "2       3.0             9\n",
       "3       4.0             6\n",
       "4       5.0             6\n",
       "..      ...           ...\n",
       "95     96.0             6\n",
       "96     97.0             7\n",
       "97     98.0             7\n",
       "98     99.0             6\n",
       "99    100.0             3\n",
       "\n",
       "[100 rows x 2 columns]"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# The user rating dataframe shows the number of ratings given with respect to the user\n",
    "user_rating = rating.groupby(by='User_ID').count()\n",
    "user_rating = user_rating['Rating'].reset_index().rename(columns={'Rating':'Rating_count'})\n",
    "user_rating"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "f1b484d5",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:54.737302Z",
     "iopub.status.busy": "2023-01-07T07:06:54.736873Z",
     "iopub.status.idle": "2023-01-07T07:06:54.749299Z",
     "shell.execute_reply": "2023-01-07T07:06:54.747992Z"
    },
    "papermill": {
     "duration": 0.034317,
     "end_time": "2023-01-07T07:06:54.752119",
     "exception": false,
     "start_time": "2023-01-07T07:06:54.717802",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count    100.000000\n",
       "mean       5.110000\n",
       "std        2.352282\n",
       "min        1.000000\n",
       "25%        3.000000\n",
       "50%        5.000000\n",
       "75%        7.000000\n",
       "max       11.000000\n",
       "Name: Rating_count, dtype: float64"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# User rating dataframe description \n",
    "user_rating[\"Rating_count\"].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "192122f6",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:54.787937Z",
     "iopub.status.busy": "2023-01-07T07:06:54.787013Z",
     "iopub.status.idle": "2023-01-07T07:06:54.842503Z",
     "shell.execute_reply": "2023-01-07T07:06:54.841291Z"
    },
    "papermill": {
     "duration": 0.076354,
     "end_time": "2023-01-07T07:06:54.845249",
     "exception": false,
     "start_time": "2023-01-07T07:06:54.768895",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th>User_ID</th>\n",
       "      <th>1.0</th>\n",
       "      <th>2.0</th>\n",
       "      <th>3.0</th>\n",
       "      <th>4.0</th>\n",
       "      <th>5.0</th>\n",
       "      <th>6.0</th>\n",
       "      <th>7.0</th>\n",
       "      <th>8.0</th>\n",
       "      <th>9.0</th>\n",
       "      <th>10.0</th>\n",
       "      <th>...</th>\n",
       "      <th>91.0</th>\n",
       "      <th>92.0</th>\n",
       "      <th>93.0</th>\n",
       "      <th>94.0</th>\n",
       "      <th>95.0</th>\n",
       "      <th>96.0</th>\n",
       "      <th>97.0</th>\n",
       "      <th>98.0</th>\n",
       "      <th>99.0</th>\n",
       "      <th>100.0</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Food_ID</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1.0</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2.0</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3.0</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4.0</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5.0</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 100 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "User_ID  1.0    2.0    3.0    4.0    5.0    6.0    7.0    8.0    9.0    10.0   \\\n",
       "Food_ID                                                                         \n",
       "1.0        0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0   \n",
       "2.0        0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    3.0    0.0   \n",
       "3.0        0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0   \n",
       "4.0        0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0   \n",
       "5.0        0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0   \n",
       "\n",
       "User_ID  ...  91.0   92.0   93.0   94.0   95.0   96.0   97.0   98.0   99.0   \\\n",
       "Food_ID  ...                                                                  \n",
       "1.0      ...    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0   \n",
       "2.0      ...    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0   \n",
       "3.0      ...    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0   \n",
       "4.0      ...    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0   \n",
       "5.0      ...    0.0    0.0    0.0    2.0    0.0    0.0    0.0    7.0    0.0   \n",
       "\n",
       "User_ID  100.0  \n",
       "Food_ID         \n",
       "1.0        0.0  \n",
       "2.0        0.0  \n",
       "3.0        0.0  \n",
       "4.0        0.0  \n",
       "5.0        0.0  \n",
       "\n",
       "[5 rows x 100 columns]"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Ultimate Table\n",
    "rating_matrix = rating.pivot_table(index='Food_ID',columns='User_ID',values='Rating').fillna(0)\n",
    "rating_matrix.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "09428c0f",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:54.880746Z",
     "iopub.status.busy": "2023-01-07T07:06:54.880283Z",
     "iopub.status.idle": "2023-01-07T07:06:54.887615Z",
     "shell.execute_reply": "2023-01-07T07:06:54.886425Z"
    },
    "papermill": {
     "duration": 0.027981,
     "end_time": "2023-01-07T07:06:54.890207",
     "exception": false,
     "start_time": "2023-01-07T07:06:54.862226",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(309, 100)"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Shape of rating_matrix\n",
    "rating_matrix.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "97ca4662",
   "metadata": {
    "papermill": {
     "duration": 0.016711,
     "end_time": "2023-01-07T07:06:54.924031",
     "exception": false,
     "start_time": "2023-01-07T07:06:54.907320",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### What is csr_matrix?\n",
    "Sparse matrix is a matrix that contains a lot of zeros. This functions helps to compress the sparse matrix. \n",
    "Thanks to https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "7e8ce6dd",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:54.960445Z",
     "iopub.status.busy": "2023-01-07T07:06:54.959309Z",
     "iopub.status.idle": "2023-01-07T07:06:54.967482Z",
     "shell.execute_reply": "2023-01-07T07:06:54.965747Z"
    },
    "papermill": {
     "duration": 0.029201,
     "end_time": "2023-01-07T07:06:54.970080",
     "exception": false,
     "start_time": "2023-01-07T07:06:54.940879",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  (0, 48)\t5.0\n",
      "  (0, 70)\t10.0\n",
      "  (1, 8)\t3.0\n",
      "  (1, 21)\t5.0\n",
      "  (1, 38)\t10.0\n",
      "  (2, 76)\t1.0\n",
      "  (2, 88)\t7.0\n",
      "  (3, 40)\t6.0\n",
      "  (3, 89)\t6.0\n",
      "  (4, 27)\t10.0\n",
      "  (4, 38)\t10.0\n",
      "  (4, 72)\t7.0\n",
      "  (4, 87)\t3.0\n",
      "  (4, 93)\t2.0\n",
      "  (4, 97)\t7.0\n",
      "  (5, 42)\t10.0\n",
      "  (5, 48)\t6.0\n",
      "  (5, 76)\t6.0\n",
      "  (5, 91)\t1.0\n",
      "  (6, 16)\t4.0\n",
      "  (6, 47)\t5.0\n",
      "  (6, 62)\t9.0\n",
      "  (6, 69)\t8.0\n",
      "  (6, 70)\t8.0\n",
      "  (7, 4)\t6.0\n",
      "  :\t:\n",
      "  (284, 30)\t9.0\n",
      "  (285, 80)\t6.0\n",
      "  (286, 24)\t3.0\n",
      "  (287, 54)\t3.0\n",
      "  (288, 55)\t9.0\n",
      "  (289, 31)\t7.0\n",
      "  (290, 15)\t1.0\n",
      "  (291, 2)\t8.0\n",
      "  (292, 95)\t5.0\n",
      "  (293, 41)\t4.0\n",
      "  (294, 43)\t10.0\n",
      "  (295, 41)\t10.0\n",
      "  (296, 94)\t5.0\n",
      "  (297, 55)\t4.0\n",
      "  (298, 2)\t1.0\n",
      "  (299, 28)\t9.0\n",
      "  (300, 53)\t1.0\n",
      "  (301, 77)\t5.0\n",
      "  (302, 63)\t6.0\n",
      "  (303, 29)\t1.0\n",
      "  (304, 55)\t9.0\n",
      "  (305, 79)\t8.0\n",
      "  (306, 70)\t1.0\n",
      "  (307, 96)\t3.0\n",
      "  (308, 31)\t5.0\n"
     ]
    }
   ],
   "source": [
    "csr_rating_matrix =  csr_matrix(rating_matrix.values)\n",
    "print(csr_rating_matrix)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4bee3d6b",
   "metadata": {
    "papermill": {
     "duration": 0.017386,
     "end_time": "2023-01-07T07:06:55.005220",
     "exception": false,
     "start_time": "2023-01-07T07:06:54.987834",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### What is Nearest Neighbours?\n",
    "Reference Link: https://scikit-learn.org/stable/modules/neighbors.html"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "6d1aaa76",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:55.043954Z",
     "iopub.status.busy": "2023-01-07T07:06:55.042983Z",
     "iopub.status.idle": "2023-01-07T07:06:55.056980Z",
     "shell.execute_reply": "2023-01-07T07:06:55.055392Z"
    },
    "papermill": {
     "duration": 0.035668,
     "end_time": "2023-01-07T07:06:55.059506",
     "exception": false,
     "start_time": "2023-01-07T07:06:55.023838",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "NearestNeighbors(metric='cosine')"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Using cosine similarity to find nearest neigbours \n",
    "recommender = NearestNeighbors(metric='cosine')\n",
    "recommender.fit(csr_rating_matrix)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "e8381ad1",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:55.098245Z",
     "iopub.status.busy": "2023-01-07T07:06:55.097011Z",
     "iopub.status.idle": "2023-01-07T07:06:55.106240Z",
     "shell.execute_reply": "2023-01-07T07:06:55.105130Z"
    },
    "papermill": {
     "duration": 0.032349,
     "end_time": "2023-01-07T07:06:55.109313",
     "exception": false,
     "start_time": "2023-01-07T07:06:55.076964",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# The main recommender code!\n",
    "def Get_Recommendations(title):\n",
    "    user= df[df['Name']==title]\n",
    "    user_index = np.where(rating_matrix.index==int(user['Food_ID']))[0][0]\n",
    "    user_ratings = rating_matrix.iloc[user_index]\n",
    "\n",
    "    reshaped = user_ratings.values.reshape(1,-1)\n",
    "    distances, indices = recommender.kneighbors(reshaped,n_neighbors=16)\n",
    "    \n",
    "    nearest_neighbors_indices = rating_matrix.iloc[indices[0]].index[1:]\n",
    "    nearest_neighbors = pd.DataFrame({'Food_ID': nearest_neighbors_indices})\n",
    "    \n",
    "    result = pd.merge(nearest_neighbors,df,on='Food_ID',how='left')\n",
    "    \n",
    "    return result.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "39167eef",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-01-07T07:06:55.148031Z",
     "iopub.status.busy": "2023-01-07T07:06:55.147549Z",
     "iopub.status.idle": "2023-01-07T07:06:55.173455Z",
     "shell.execute_reply": "2023-01-07T07:06:55.172210Z"
    },
    "papermill": {
     "duration": 0.047911,
     "end_time": "2023-01-07T07:06:55.176128",
     "exception": false,
     "start_time": "2023-01-07T07:06:55.128217",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Food_ID</th>\n",
       "      <th>index</th>\n",
       "      <th>Name</th>\n",
       "      <th>C_Type</th>\n",
       "      <th>Veg_Non</th>\n",
       "      <th>Describe</th>\n",
       "      <th>soup</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>126.0</td>\n",
       "      <td>125</td>\n",
       "      <td>andhra crab meat masala</td>\n",
       "      <td>Indian</td>\n",
       "      <td>non-veg</td>\n",
       "      <td>processed crab meat refined oil curry leaves g...</td>\n",
       "      <td>Indian non-veg processed crab meat refined oil...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>75.0</td>\n",
       "      <td>74</td>\n",
       "      <td>detox haldi tea</td>\n",
       "      <td>Beverage</td>\n",
       "      <td>veg</td>\n",
       "      <td>haldi ginger black pepper honey water</td>\n",
       "      <td>Beverage veg haldi ginger black pepper honey w...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>100.0</td>\n",
       "      <td>99</td>\n",
       "      <td>spicy chicken curry</td>\n",
       "      <td>Indian</td>\n",
       "      <td>non-veg</td>\n",
       "      <td>oil ghee onion paste garlic paste ginger paste...</td>\n",
       "      <td>Indian non-veg oil ghee onion paste garlic pas...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>259.0</td>\n",
       "      <td>258</td>\n",
       "      <td>ragi coconut ladoo (laddu)</td>\n",
       "      <td>Dessert</td>\n",
       "      <td>veg</td>\n",
       "      <td>finger millet flour ragi jaggery peanuts cocon...</td>\n",
       "      <td>Dessert veg finger millet flour ragi jaggery p...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>51.0</td>\n",
       "      <td>50</td>\n",
       "      <td>christmas chocolate fudge cookies</td>\n",
       "      <td>Dessert</td>\n",
       "      <td>veg</td>\n",
       "      <td>unsalted butter brown sugar chocolate chocolat...</td>\n",
       "      <td>Dessert veg unsalted butter brown sugar chocol...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Food_ID  index                               Name    C_Type  Veg_Non  \\\n",
       "0    126.0    125            andhra crab meat masala    Indian  non-veg   \n",
       "1     75.0     74                    detox haldi tea  Beverage      veg   \n",
       "2    100.0     99                spicy chicken curry    Indian  non-veg   \n",
       "3    259.0    258         ragi coconut ladoo (laddu)   Dessert      veg   \n",
       "4     51.0     50  christmas chocolate fudge cookies   Dessert      veg   \n",
       "\n",
       "                                            Describe  \\\n",
       "0  processed crab meat refined oil curry leaves g...   \n",
       "1              haldi ginger black pepper honey water   \n",
       "2  oil ghee onion paste garlic paste ginger paste...   \n",
       "3  finger millet flour ragi jaggery peanuts cocon...   \n",
       "4  unsalted butter brown sugar chocolate chocolat...   \n",
       "\n",
       "                                                soup  \n",
       "0  Indian non-veg processed crab meat refined oil...  \n",
       "1  Beverage veg haldi ginger black pepper honey w...  \n",
       "2  Indian non-veg oil ghee onion paste garlic pas...  \n",
       "3  Dessert veg finger millet flour ragi jaggery p...  \n",
       "4  Dessert veg unsalted butter brown sugar chocol...  "
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Get recommendations with this function \n",
    "Get_Recommendations('tricolour salad')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "db9b7346",
   "metadata": {
    "papermill": {
     "duration": 0.017754,
     "end_time": "2023-01-07T07:06:55.211246",
     "exception": false,
     "start_time": "2023-01-07T07:06:55.193492",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Upvote if you like this ✨"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.12"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 14.847681,
   "end_time": "2023-01-07T07:06:56.053253",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2023-01-07T07:06:41.205572",
   "version": "2.3.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
