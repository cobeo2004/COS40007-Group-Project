import Image, { type ImageProps } from "next/image";
import { getQueryClient } from "../utils/server";
import { sayHelloGetOptions } from "../lib/open-api/@tanstack/react-query.gen";
import dynamic from "next/dynamic";

// Dynamically import the client component with no SSR
const MediaUpload = dynamic(() => import("./_components/media-upload"), {
  ssr: true,
});

type Props = Omit<ImageProps, "src"> & {
  srcLight: string;
  srcDark: string;
};

const ThemeImage = (props: Props) => {
  const { srcLight, srcDark, ...rest } = props;

  return (
    <>
      <Image {...rest} src={srcLight} className="imgLight" />
      <Image {...rest} src={srcDark} className="imgDark" />
    </>
  );
};

export default async function Home() {
  const qc = getQueryClient();
  await qc.prefetchQuery(sayHelloGetOptions());
  return (
    <div className="min-h-screen bg-white">
      <main className="container mx-auto py-8 px-4">
        <div className="bg-white rounded-lg shadow-md p-6">
          <MediaUpload />
        </div>
      </main>
    </div>
  );
}
