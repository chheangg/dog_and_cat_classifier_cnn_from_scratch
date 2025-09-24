import { cn } from "../lib/utils";
import { DynamicIcon, type IconName } from 'lucide-react/dynamic';

// from https://github.com/neobrutalism-templates/bento/blob/main/src/app/page.tsx
export function ActionButton(
  { className, title, subtitle, icon, onClick  } :
  {
    className?: string,
    title: string,
    subtitle: string,
    icon: IconName,
    onClick?: () => void;
  }
) {
  return (
    <a
      className={cn([
        "border-border flex flex-col w-48 max-w-48 md:w-60 md:max-w-60 shadow-shadow aspect-square text-main-foreground rounded-base bg-main cursor-pointer hover:translate-x-boxShadowX hover:translate-y-boxShadowY border-2 p-5 transition-all hover:shadow-none",
        className
      ])}
      onClick={onClick}
    >
      <DynamicIcon className="mt-auto w-8 sm:w-10 lg:w-12 h-8 sm:h-10 lg:h-12" name={icon} />
      <p className="mt-3 font-heading text-md sm:text-xl lg:text-2xl">
        {title}
      </p>
      <p className="mt-1 font-base text-xs sm:text-sm lg:text-base">
        {subtitle}
      </p>
    </a>
  )
}