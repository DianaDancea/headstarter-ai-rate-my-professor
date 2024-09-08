import { NextResponse } from 'next/server';
import { OpenAI } from 'openai';

// Import PineconeClient using CommonJS syntax
const { Pinecone: PineconeClient } = require('@pinecone-database/pinecone');

// Initialize Pinecone client
const pinecone = new PineconeClient({
  apiKey: process.env.PINECONE_API_KEY,
});

const index = pinecone.Index('rag'); // Replace 'rag' with your actual index name
const openai = new OpenAI();

// Define the system prompt
const systemPrompt = `
You are a rate my professor agent to help students find classes, that takes in user questions and answers them.
For every user question, the top 3 professors that match the user question are returned.
Use them to answer the question if needed.
`;

export async function POST(req) {
  const data = await req.json();

  try {
    const text = data[data.length - 1].content;
    const embeddingResponse = await openai.embeddings.create({
      model: 'text-embedding-3-small',
      input: text,
    });

    const embedding = embeddingResponse.data[0].embedding;

    const results = await index.query({
      topK: 5,
      includeMetadata: true,
      vector: embedding,
    });

    let resultString = '';
    results.matches.forEach((match) => {
      resultString += `
      Returned Results:
      Professor: ${match.id}
      Review: ${match.metadata.review}
      Subject: ${match.metadata.subject}
      Stars: ${match.metadata.stars}
      \n\n`;
    });

    const lastMessage = data[data.length - 1];
    const lastMessageContent = lastMessage.content + resultString;
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1);

    const completion = await openai.chat.completions.create({
      messages: [
        { role: 'system', content: systemPrompt },
        ...lastDataWithoutLastMessage,
        { role: 'user', content: lastMessageContent },
      ],
      model: 'gpt-3.5-turbo',
      stream: true,
    });

    const stream = new ReadableStream({
      async start(controller) {
        const encoder = new TextEncoder();
        try {
          for await (const chunk of completion) {
            const content = chunk.choices[0]?.delta?.content;
            if (content) {
              const text = encoder.encode(content);
              controller.enqueue(text);
            }
          }
        } catch (err) {
          controller.error(err);
        } finally {
          controller.close();
        }
      },
    });

    return new NextResponse(stream);

  } catch (error) {
    console.error('Error processing request:', error);
    return new NextResponse('Internal Server Error', { status: 500 });
  }
}
