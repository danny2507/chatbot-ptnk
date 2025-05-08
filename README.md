## Installation

1. Clone the repository:
   ```
   git clone https://github.com/danny2507/chatbot-ptnk.git
   cd chatbot-ptnk
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   - Create a Hugging Face account and get your access token
   - Rename `.envsample` to `.env`
   - Add your Hugging Face token to the `.env` file:
     ```
     HUGGINGFACE_TOKEN=your_token_here
     ```

## Usage

To run the application with hot-reloading enabled:

```
chainlit run chat.py -w
```

This will start the Chainlit server and automatically reload whenever you make changes to the code.